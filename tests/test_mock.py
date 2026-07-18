"""
Torch-free self-tests for the parts of the pipeline that don't need a GPU:
the differential beam-search mechanics and the metric battery.

A ``MockOracle`` fabricates logits so that a chosen "hidden" token sequence has
a large target-vs-base delta at each position, while all other tokens have
delta ~ 0. A correct differential search must recover that sequence. This
verifies the search logic (token-id handling, thresholding, backtracking,
stopping) independently of any model.

Run: ``python tests/test_mock.py``  (needs only numpy).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics
from attack import DifferentialExtractor
from metrics import QueryCounter


VOCAB = 50
CLOSE_ID = 49  # stands in for </think>


class MockOracle:
    """Deterministic oracle: the hidden sequence is primed by a +delta bump.

    target = base + bump, where bump[next_hidden_token] = primed_delta and every
    other token gets ~0. base is a fixed pseudo-random LM prior. This is exactly
    the structure the attack assumes: hidden-context tokens spike in
    (target - base), LM artifacts cancel.
    """

    def __init__(self, hidden, primed_delta=5.0, noise=0.05):
        self.hidden = list(hidden)
        self.primed_delta = primed_delta
        self.noise = noise
        self.counter = QueryCounter()
        rng = np.random.default_rng(0)
        # A stable "language-model prior" indexed by prefix length.
        self._priors = [rng.normal(size=VOCAB) for _ in range(len(self.hidden) + 2)]

    @property
    def think_close_id(self):
        return CLOSE_ID

    def _prior(self, prefix):
        return self._priors[min(len(prefix), len(self._priors) - 1)]

    def base_logits(self, prefix):
        self.counter.tick()
        return self._prior(prefix).copy()

    def target_logits(self, prefix):
        self.counter.tick()
        logits = self._prior(prefix).copy()
        pos = len(prefix)
        if pos < len(self.hidden):
            logits[self.hidden[pos]] += self.primed_delta
        else:
            logits[CLOSE_ID] += self.primed_delta  # spike the stop token at the end
        return logits


def test_beam_recovers_hidden_sequence():
    hidden = [3, 14, 7, 42, 8]
    oracle = MockOracle(hidden, primed_delta=5.0)
    extractor = DifferentialExtractor(
        oracle, beam_width=4, top_k=6, delta_threshold=1.0, max_tokens=20
    )
    result = extractor.run(decode=lambda ids: ",".join(map(str, ids)))
    assert result.token_ids == hidden, f"got {result.token_ids}, want {hidden}"
    # Two forwards (target+base) per beam per step -> bounded, > 0.
    assert result.forward_passes > 0
    assert result.queries_per_token() > 0
    print(f"  beam recovered hidden ids exactly in {result.forward_passes} passes "
          f"({result.queries_per_token():.1f} q/token)")


def test_backtracking_stops_on_flat_signal():
    # primed_delta below threshold everywhere -> no anomalous spikes -> the
    # search must refuse to hallucinate a long sequence.
    oracle = MockOracle([1, 2, 3], primed_delta=0.2)
    extractor = DifferentialExtractor(
        oracle, beam_width=3, top_k=5, delta_threshold=1.0, max_tokens=20
    )
    result = extractor.run()
    assert len(result.token_ids) == 0, f"expected empty extraction, got {result.token_ids}"
    print("  flat signal correctly yields no extraction (backtracking works)")


def test_metrics_sanity():
    truth = list(range(20))
    assert metrics.exact_match(truth, truth)
    assert metrics.verbatim_overlap(truth, truth) == 1.0
    assert metrics.normalized_levenshtein_similarity(truth, truth) == 1.0
    assert metrics.token_f1(truth, truth) == 1.0
    # Half-overlap prediction.
    pred = list(range(10)) + list(range(100, 110))
    assert 0.0 < metrics.verbatim_overlap(pred, truth) < 1.0
    assert 0.0 < metrics.token_f1(pred, truth) < 1.0
    assert metrics.ngram_extraction_rate(truth, truth, n=5) == 1.0
    assert metrics.ngram_extraction_rate(truth, truth, n=50) is None  # too short
    # Text-level metrics degrade gracefully when optional libs are absent.
    scores = metrics.evaluate("the cat sat", "the cat sat")
    assert scores.exact_match
    print("  metric battery sane (exact/verbatim/edit/f1/ngram + graceful optionals)")


def test_query_counter():
    c = QueryCounter()
    c.tick(); c.tick(3)
    assert c.forward_passes == 4
    assert c.per_token(2) == 2.0
    print("  query counter accounts forward passes correctly")


if __name__ == "__main__":
    tests = [
        test_beam_recovers_hidden_sequence,
        test_backtracking_stops_on_flat_signal,
        test_metrics_sanity,
        test_query_counter,
    ]
    for t in tests:
        print(f"[test] {t.__name__}")
        t()
    print("\nAll mock self-tests passed.")
