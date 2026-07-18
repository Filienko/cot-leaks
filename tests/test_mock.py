"""
Torch-free self-tests for the GPU-independent logic: the differential
beam-search mechanics (both methods) and the metric battery.

A ``MockOracle`` fabricates a hidden token sequence and the two signals the
attack relies on:

  * answer likelihood  -- log P(A | reasoning) grows by a fixed bonus for every
    correctly recovered prefix token of the hidden sequence, and not at all once
    a path diverges. A correct answer-anchored search must climb that signal to
    recover the hidden sequence and stop when it goes flat.
  * question-priming delta -- target-base logits bump the correct next token, so
    candidate proposal surfaces it.

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


def _match_len(prefix, hidden):
    k = 0
    while k < len(hidden) and k < len(prefix) and prefix[k] == hidden[k]:
        k += 1
    return k


class MockOracle:
    """Deterministic oracle whose answer likelihood rewards recovering `hidden`."""

    def __init__(self, hidden, bonus=2.0, primed_delta=6.0):
        self.hidden = list(hidden)
        self.bonus = bonus
        self.primed_delta = primed_delta
        self.counter = QueryCounter()
        rng = np.random.default_rng(0)
        self._priors = [rng.normal(size=VOCAB) for _ in range(len(self.hidden) + 3)]

    @property
    def think_close_id(self):
        return CLOSE_ID

    def _prior(self, prefix):
        return self._priors[min(len(prefix), len(self._priors) - 1)].copy()

    # -- proposal signals -------------------------------------------------- #
    def base_logits(self, prefix):
        self.counter.tick()
        return self._prior(prefix)

    def target_logits(self, prefix):
        self.counter.tick()
        logits = self._prior(prefix)
        m = _match_len(prefix, self.hidden)
        if m < len(self.hidden):
            logits[self.hidden[m]] += self.primed_delta  # bump the correct next token
        return logits

    # -- answer-anchored scoring ------------------------------------------- #
    def answer_loglik_batch(self, prefixes):
        self.counter.tick(len(prefixes))
        return [self.bonus * _match_len(p, self.hidden) for p in prefixes]

    def base_answer_loglik(self):
        return 0.0


def test_answer_anchored_recovers_hidden():
    hidden = [3, 14, 7, 42, 8]
    oracle = MockOracle(hidden)
    extractor = DifferentialExtractor(
        oracle, method="answer_anchored", beam_width=4, top_k=6,
        gain_threshold=0.5, max_tokens=20,
    )
    result = extractor.run(decode=lambda ids: ",".join(map(str, ids)))
    assert result.token_ids == hidden, f"got {result.token_ids}, want {hidden}"
    assert result.forward_passes > 0
    # delta log P(A) = bonus * len(hidden).
    assert abs(result.score - 2.0 * len(hidden)) < 1e-6
    print(f"  answer-anchored recovered hidden exactly; delta logP(A)={result.score:.1f}, "
          f"{result.queries_per_token():.1f} q/token")


def test_answer_anchored_stops_on_flat_signal():
    # bonus below threshold -> no token meaningfully explains A -> stop early.
    oracle = MockOracle([1, 2, 3], bonus=0.1)
    extractor = DifferentialExtractor(
        oracle, method="answer_anchored", beam_width=3, top_k=5,
        gain_threshold=0.5, max_tokens=20,
    )
    result = extractor.run()
    assert len(result.token_ids) == 0, f"expected empty, got {result.token_ids}"
    print("  flat answer signal correctly yields no extraction (backtracking works)")


def test_continuation_recovers_hidden():
    hidden = [5, 11, 30, 2]
    oracle = MockOracle(hidden, primed_delta=6.0)
    extractor = DifferentialExtractor(
        oracle, method="continuation", beam_width=4, top_k=6,
        delta_threshold=1.0, max_tokens=20,
    )
    result = extractor.run()
    assert result.token_ids == hidden, f"got {result.token_ids}, want {hidden}"
    print(f"  continuation method also recovers hidden via priming delta "
          f"({result.forward_passes} passes)")


def test_metrics_sanity():
    truth = list(range(20))
    assert metrics.exact_match(truth, truth)
    assert metrics.verbatim_overlap(truth, truth) == 1.0
    assert metrics.normalized_levenshtein_similarity(truth, truth) == 1.0
    assert metrics.token_f1(truth, truth) == 1.0
    pred = list(range(10)) + list(range(100, 110))
    assert 0.0 < metrics.verbatim_overlap(pred, truth) < 1.0
    assert 0.0 < metrics.token_f1(pred, truth) < 1.0
    assert metrics.ngram_extraction_rate(truth, truth, n=5) == 1.0
    assert metrics.ngram_extraction_rate(truth, truth, n=50) is None
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
        test_answer_anchored_recovers_hidden,
        test_answer_anchored_stops_on_flat_signal,
        test_continuation_recovers_hidden,
        test_metrics_sanity,
        test_query_counter,
    ]
    for t in tests:
        print(f"[test] {t.__name__}")
        t()
    print("\nAll mock self-tests passed.")
