"""
Differential logit side-channel attack for reconstructing a reasoning model's
hidden chain-of-thought (CoT).

Core idea (the project's thesis)
--------------------------------
Holding a hidden context in the window leaves a measurable, anomalous *upward*
shift in the logits of the tokens that make up that context, beyond background
language-modelling noise. We isolate that shift with a **differential** query:

    delta = logits_target(context + prefix)  -  logits_base(prefix)

where
  * the *target* instance is primed with the hidden context (here: the question
    the victim reasoned about, sitting in front of the hidden <think> trace), and
  * the *base* instance is a zero-context control -- the same model, same
    reconstructed prefix, but with the priming context removed.

Tokens with ``delta ~ 0`` are ordinary LM continuations and are pruned; tokens
with ``delta`` above a threshold are flagged as *context-primed* and become beam
candidates. We reconstruct the hidden CoT autoregressively over **token ids**
(never over decoded strings -- see the tokenization-drift note below), appending
the highest-delta token, advancing to the next position, and backtracking when a
path goes flat (no anomalous spike = wrong turn).

Why token ids, not strings
--------------------------
BPE is not round-trip safe: ``decode([id])`` followed by re-``encode`` of the
concatenated string can land on a different id sequence (leading-space / byte
tokens). Scoring on re-tokenised strings therefore silently corrupts the logit
comparison. We keep an explicit id list and only ever *append* ids to it.

Decoupling
----------
The beam search consumes plain numpy logits through a small ``LogitOracle``
interface, so the search logic is unit-testable without torch (see
``tests/test_mock.py``). ``HFOracle`` is the real implementation backed by a
Hugging Face model on GPU.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from metrics import QueryCounter


# --------------------------------------------------------------------------- #
# Oracle interface
# --------------------------------------------------------------------------- #
class LogitOracle(Protocol):
    """Abstraction over the two model instances the attack can query."""

    def target_logits(self, prefix_ids: Sequence[int]) -> np.ndarray:
        """Next-token logits for (hidden context + reconstructed prefix)."""

    def base_logits(self, prefix_ids: Sequence[int]) -> np.ndarray:
        """Next-token logits for (reconstructed prefix) with no priming context."""

    @property
    def think_close_id(self) -> int:
        """Token id of ``</think>`` -- the natural stop signal for the CoT."""


# --------------------------------------------------------------------------- #
# Hugging Face oracle (real target, runs on GPU)
# --------------------------------------------------------------------------- #
class HFOracle:
    """Backs the differential attack with a real reasoning model.

    ``context_ids`` are the token ids that prime the target instance: the
    chat-templated question followed by the opening ``<think>`` tag. The base
    instance uses ``base_context_ids`` -- by default just the opening
    ``<think>`` tag, i.e. the same start-of-reasoning state with the question
    (the hidden context) removed, isolating the question's priming effect.

    Every forward pass is counted through ``counter`` for the queries/token
    efficiency metric. Candidate continuations that share a prefix are batched
    into a single forward pass; the delta at the current position already ranks
    the whole vocabulary, so we need only two forwards (target + base) per beam
    per step -- not one per candidate token as in the original PoC.
    """

    def __init__(self, model, tokenizer, context_ids, base_context_ids=None,
                 counter: Optional[QueryCounter] = None):
        import torch  # local import so metrics/tests stay torch-free

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.context_ids = list(context_ids)
        # Zero-context control: keep the <think> opener, drop the question.
        if base_context_ids is None:
            base_context_ids = _trailing_think_open(tokenizer)
        self.base_context_ids = list(base_context_ids)
        self.counter = counter or QueryCounter()

        close = tokenizer.encode("</think>", add_special_tokens=False)
        # Some tokenizers make </think> a single special token; take the last id.
        self._think_close_id = close[-1] if close else tokenizer.eos_token_id

    @property
    def think_close_id(self) -> int:
        return self._think_close_id

    def _logits(self, ids: Sequence[int]) -> np.ndarray:
        torch = self.torch
        input_ids = torch.tensor([list(ids)], device=self.device)
        with torch.no_grad():
            out = self.model(input_ids)
        self.counter.tick()
        # bf16/fp16 -> float32 numpy for stable, torch-free downstream math.
        return out.logits[0, -1, :].float().cpu().numpy()

    def target_logits(self, prefix_ids: Sequence[int]) -> np.ndarray:
        return self._logits(self.context_ids + list(prefix_ids))

    def base_logits(self, prefix_ids: Sequence[int]) -> np.ndarray:
        return self._logits(self.base_context_ids + list(prefix_ids))


def _trailing_think_open(tokenizer) -> List[int]:
    ids = tokenizer.encode("<think>\n", add_special_tokens=False)
    return ids if ids else []


# --------------------------------------------------------------------------- #
# Differential beam search
# --------------------------------------------------------------------------- #
@dataclass(order=True)
class Beam:
    # heapq is a min-heap; we store negative score so the best beam pops first.
    neg_score: float
    seq: Tuple[int, ...] = field(compare=False, default=())
    finished: bool = field(compare=False, default=False)
    dead: bool = field(compare=False, default=False)  # flat-signal wrong turn


@dataclass
class ExtractionResult:
    token_ids: List[int]
    text: str
    forward_passes: int
    steps: int
    per_position_max_delta: List[float]

    def queries_per_token(self) -> float:
        n = len(self.token_ids) or 1
        return self.forward_passes / n


class DifferentialExtractor:
    """Autoregressive backtracking beam search driven by target-base deltas."""

    def __init__(
        self,
        oracle: LogitOracle,
        beam_width: int = 4,
        top_k: int = 8,
        delta_threshold: float = 1.0,
        max_tokens: int = 64,
    ):
        self.oracle = oracle
        self.beam_width = beam_width
        self.top_k = top_k
        self.delta_threshold = delta_threshold
        self.max_tokens = max_tokens

    def _expand(self, beam: Beam):
        """Return candidate (delta_value, token_id) pairs for one beam step."""
        tgt = self.oracle.target_logits(beam.seq)
        base = self.oracle.base_logits(beam.seq)
        delta = tgt - base  # anomalous upward shift = context priming
        # Top-k tokens by *delta*, not by raw target probability: this is the
        # pruning rule from the design -- discard normal LM artifacts (delta~0),
        # keep context-primed spikes.
        k = min(self.top_k, delta.shape[0])
        idx = np.argpartition(delta, -k)[-k:]
        idx = idx[np.argsort(delta[idx])[::-1]]  # descending by delta
        return [(float(delta[i]), int(i)) for i in idx]

    def run(self, decode=None) -> ExtractionResult:
        """Execute the search. ``decode`` maps an id list -> text for the result."""
        beams: List[Beam] = [Beam(neg_score=0.0, seq=(), finished=False)]
        per_position_max_delta: List[float] = []
        start_fp = _counter_value(self.oracle)

        for step in range(self.max_tokens):
            if all(b.finished for b in beams):
                break

            candidates: List[Beam] = []
            step_max_delta = 0.0

            for b in beams:
                if b.finished:
                    candidates.append(b)
                    continue

                expansions = self._expand(b)
                best_delta = expansions[0][0] if expansions else 0.0
                step_max_delta = max(step_max_delta, best_delta)

                # Backtracking rule: a flat signal (no token clears the
                # threshold) means this path is a wrong turn. We do not extend
                # it; other beams -- i.e. the next-best anomalies from earlier
                # steps -- carry the search forward instead.
                if best_delta < self.delta_threshold:
                    b.dead = True
                    b.finished = True
                    candidates.append(b)
                    continue

                for delta_val, token_id in expansions:
                    if delta_val < self.delta_threshold:
                        break  # expansions are sorted; rest are below threshold
                    seq = b.seq + (token_id,)
                    # Reward high-delta tokens; mean delta keeps scores
                    # comparable across paths of different length (fixes the
                    # cumulative-sum length bias in the original PoC).
                    new_score = (-b.neg_score * len(b.seq) + delta_val) / len(seq)
                    finished = token_id == self.oracle.think_close_id
                    candidates.append(
                        Beam(neg_score=-new_score, seq=seq, finished=finished)
                    )

            per_position_max_delta.append(step_max_delta)
            # Keep the best `beam_width` live+finished beams.
            beams = heapq.nsmallest(self.beam_width, candidates)

        # Prefer a finished, non-dead beam; else the highest-scoring live one.
        ranked = sorted(beams, key=lambda b: (b.dead, b.neg_score))
        best = ranked[0]
        token_ids = [t for t in best.seq if t != self.oracle.think_close_id]
        text = decode(token_ids) if decode else ""
        forward_passes = _counter_value(self.oracle) - start_fp

        return ExtractionResult(
            token_ids=token_ids,
            text=text,
            forward_passes=forward_passes,
            steps=len(per_position_max_delta),
            per_position_max_delta=per_position_max_delta,
        )


def _counter_value(oracle) -> int:
    counter = getattr(oracle, "counter", None)
    return counter.forward_passes if counter is not None else 0


# --------------------------------------------------------------------------- #
# Convenience builder for the CoT threat model
# --------------------------------------------------------------------------- #
def build_cot_oracle(model, tokenizer, question: str, counter: Optional[QueryCounter] = None):
    """Construct an HFOracle primed with the victim's question.

    Target context = chat_template(question) + "<think>\\n"  (the state the model
    is in just before it starts its hidden reasoning). Base context = "<think>\\n"
    alone, so ``delta`` isolates the question's priming influence on the trace.
    """
    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Solve the following problem."},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    open_think = "<think>\n" if "<think>" not in prompt else ""
    context_ids = tokenizer.encode(prompt + open_think, add_special_tokens=False)
    base_context_ids = _trailing_think_open(tokenizer)
    return HFOracle(model, tokenizer, context_ids, base_context_ids, counter=counter)
