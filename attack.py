"""
Differential logit side-channel attack for reconstructing a reasoning model's
hidden chain-of-thought (CoT).

Threat model
------------
The attacker knows the user's **question Q** and the model's **public answer A**
(the output that is *not* between the <think> tags). The attacker does NOT know
the hidden reasoning between <think> ... </think>; that is what we reconstruct.
The attacker can query the model's next-token logits (open weights: full vocab).

Two methods
-----------
1. ``answer_anchored`` (default). The public answer A is the anchor. The beam is
   *driven* by the question-priming differential (as in ``continuation``, below),
   which robustly surfaces the next reasoning token; A is then used to pick where
   to truncate. Over the primed candidates we score

        delta(c) = log P(A | Q, <think> c </think>) - log P(A | Q, <think></think>)
                            └── candidate reasoning ──┘        └── empty control ──┘

   The second term is the **zero-context / empty-reasoning** control: the answer's
   likelihood with no reasoning at all. Subtracting it isolates the marginal
   contribution of the reasoning. We emit the prefix that maximises ``delta(c)``,
   provided that maximum clears ``gain_threshold`` nats over the control (else the
   reasoning does not explain A and we emit nothing). Crucially we do NOT rank the
   beam by ``delta(c)`` per token: one reasoning token barely moves P(A) -- the
   marginal is ~0 and often negative near the root -- so per-token answer ranking
   abandons the true path in that dip. Priming drives; the answer anchors. This
   uses exactly what the attacker has (Q and A) and never touches the <think>
   tokens.

2. ``continuation`` (cheap proposal-style variant). delta = logits_target(Q +
   <think> + prefix) - logits_base(<think> + prefix) isolates the question's
   priming of the next reasoning token. Fast (2 forwards/step) but, on its own,
   it regenerates rather than inverts, since the attacker already holds Q.

Candidate proposal (both methods) ranks tokens by the question-priming delta
``logits_target - logits_base`` so we only ever score context-relevant tokens.

Why token ids, not strings
--------------------------
BPE is not round-trip safe: ``decode([id])`` then re-``encode`` of the joined
string can land on a different id sequence (leading-space / byte tokens), which
silently corrupts every logit comparison. We keep an explicit id list and only
ever *append* ids to it.

Decoupling
----------
The search consumes plain numpy logits / floats through a small ``LogitOracle``
interface, so it is unit-testable without torch (see ``tests/test_mock.py``).
``HFOracle`` is the real GPU-backed implementation.
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
    """Everything the search needs from the model instances."""

    def target_logits(self, prefix_ids: Sequence[int]) -> np.ndarray:
        """Next-token logits for (Q + <think> + reconstructed prefix)."""

    def base_logits(self, prefix_ids: Sequence[int]) -> np.ndarray:
        """Next-token logits for (<think> + prefix) with the question removed."""

    def answer_loglik_batch(self, prefixes: Sequence[Sequence[int]]) -> List[float]:
        """log P(A | Q, <think> prefix </think>) for each prefix (same length)."""

    def base_answer_loglik(self) -> float:
        """log P(A | Q, <think></think>) -- the empty-reasoning control."""

    @property
    def think_close_id(self) -> int: ...


# --------------------------------------------------------------------------- #
# Hugging Face oracle (real target, runs on GPU)
# --------------------------------------------------------------------------- #
class HFOracle:
    """Backs the attack with a real reasoning model.

    ``context_ids``      = chat_template(Q) + "<think>\\n"   (question-primed).
    ``base_context_ids`` = "<think>\\n"                       (question removed).
    ``close_ids``        = "</think>\\n\\n"  tokens between the trace and answer.
    ``answer_ids``       = the public answer A (known to the attacker).

    Every forward pass is counted for the queries/token efficiency metric. Answer
    likelihoods are computed by teacher forcing over A in a single batched
    forward per beam step (all same-length candidates in one pass).
    """

    def __init__(self, model, tokenizer, context_ids, answer_ids,
                 base_context_ids=None, close_ids=None,
                 counter: Optional[QueryCounter] = None):
        import torch  # local import so metrics/tests stay torch-free

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.context_ids = list(context_ids)
        self.answer_ids = list(answer_ids)
        self.base_context_ids = list(
            base_context_ids if base_context_ids is not None else _think_open(tokenizer)
        )
        self.close_ids = list(
            close_ids if close_ids is not None
            else tokenizer.encode("</think>\n\n", add_special_tokens=False)
        )
        self.counter = counter or QueryCounter()
        self._base_answer_ll: Optional[float] = None

        close = tokenizer.encode("</think>", add_special_tokens=False)
        self._think_close_id = close[-1] if close else tokenizer.eos_token_id

    @property
    def think_close_id(self) -> int:
        return self._think_close_id

    # -- proposal signals (single-position logits) ------------------------- #
    def _last_logits(self, ids: Sequence[int]) -> np.ndarray:
        torch = self.torch
        input_ids = torch.tensor([list(ids)], device=self.device)
        with torch.no_grad():
            out = self.model(input_ids)
        self.counter.tick()
        return out.logits[0, -1, :].float().cpu().numpy()

    def target_logits(self, prefix_ids: Sequence[int]) -> np.ndarray:
        return self._last_logits(self.context_ids + list(prefix_ids))

    def base_logits(self, prefix_ids: Sequence[int]) -> np.ndarray:
        return self._last_logits(self.base_context_ids + list(prefix_ids))

    # -- answer-anchored scoring (teacher-forced log-likelihood of A) ------- #
    def _score_seqs(self, reasoning_prefixes: Sequence[Sequence[int]]) -> List[float]:
        """log P(A | context + prefix + close) for each (equal-length) prefix."""
        torch = self.torch
        seqs = [
            self.context_ids + list(p) + self.close_ids + self.answer_ids
            for p in reasoning_prefixes
        ]
        input_ids = torch.tensor(seqs, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits  # [B, T, V]
        self.counter.tick(len(seqs))
        logp = torch.log_softmax(logits.float(), dim=-1)

        # Answer tokens occupy positions [start, start + |A|); the logits that
        # predict them are at [start-1, start-1 + |A|).
        start = len(self.context_ids) + len(reasoning_prefixes[0]) + len(self.close_ids)
        ans = torch.tensor(self.answer_ids, device=self.device)
        pred_pos = torch.arange(start - 1, start - 1 + len(self.answer_ids), device=self.device)
        sel = logp[:, pred_pos, :]  # [B, |A|, V]
        gathered = sel.gather(-1, ans.view(1, -1, 1).expand(sel.size(0), -1, 1)).squeeze(-1)
        return gathered.sum(dim=1).cpu().tolist()

    def answer_loglik_batch(self, prefixes: Sequence[Sequence[int]]) -> List[float]:
        return self._score_seqs(prefixes)

    def base_answer_loglik(self) -> float:
        if self._base_answer_ll is None:
            self._base_answer_ll = self._score_seqs([[]])[0]
        return self._base_answer_ll


def _think_open(tokenizer) -> List[int]:
    return tokenizer.encode("<think>\n", add_special_tokens=False)


# --------------------------------------------------------------------------- #
# Beam / result containers
# --------------------------------------------------------------------------- #
@dataclass(order=True)
class Beam:
    # heapq is a min-heap; we store -score so the best beam pops first.
    neg_score: float
    seq: Tuple[int, ...] = field(compare=False, default=())
    finished: bool = field(compare=False, default=False)
    dead: bool = field(compare=False, default=False)
    adelta: float = field(compare=False, default=0.0)  # cumulative delta log P(A)


@dataclass
class ExtractionResult:
    token_ids: List[int]
    text: str
    forward_passes: int
    steps: int
    score: float                       # final delta log-likelihood of A (nats)
    per_position_signal: List[float]   # per-step best signal (gain or delta)
    method: str

    def queries_per_token(self) -> float:
        n = len(self.token_ids) or 1
        return self.forward_passes / n


# --------------------------------------------------------------------------- #
# Differential extractor
# --------------------------------------------------------------------------- #
class DifferentialExtractor:
    """Backtracking beam search; answer-anchored by default."""

    def __init__(
        self,
        oracle: LogitOracle,
        method: str = "answer_anchored",
        beam_width: int = 4,
        top_k: int = 8,
        delta_threshold: float = 1.0,   # both: min priming logit delta to keep exploring
        gain_threshold: float = 0.05,   # answer_anchored: min cumulative nat gain in log P(A) to emit
        max_tokens: int = 64,
    ):
        assert method in ("answer_anchored", "continuation")
        self.oracle = oracle
        self.method = method
        self.beam_width = beam_width
        self.top_k = top_k
        self.delta_threshold = delta_threshold
        self.gain_threshold = gain_threshold
        self.max_tokens = max_tokens

    # -- candidate proposal (shared) --------------------------------------- #
    def _propose(self, seq) -> List[Tuple[float, int]]:
        """Top-k next tokens ranked by question-priming delta (target - base)."""
        tgt = self.oracle.target_logits(seq)
        base = self.oracle.base_logits(seq)
        delta = tgt - base
        k = min(self.top_k, delta.shape[0])
        idx = np.argpartition(delta, -k)[-k:]
        idx = idx[np.argsort(delta[idx])[::-1]]
        return [(float(delta[i]), int(i)) for i in idx]

    def run(self, decode=None) -> ExtractionResult:
        start_fp = _counter_value(self.oracle)
        if self.method == "answer_anchored":
            beams, signal = self._run_answer_anchored()
        else:
            beams, signal = self._run_continuation()

        ranked = sorted(beams, key=lambda b: (b.dead, b.neg_score))
        best = ranked[0]
        token_ids = [t for t in best.seq if t != self.oracle.think_close_id]
        return ExtractionResult(
            token_ids=token_ids,
            text=decode(token_ids) if decode else "",
            forward_passes=_counter_value(self.oracle) - start_fp,
            steps=len(signal),
            score=-best.neg_score,
            per_position_signal=signal,
            method=self.method,
        )

    # -- answer-anchored search -------------------------------------------- #
    def _run_answer_anchored(self):
        """Priming-driven reconstruction with answer-anchored truncation.

        A single reasoning token barely shifts the likelihood of the whole answer
        A (and can even *lower* it, since a one-word-then-closed trace is an odd
        state), so the per-token answer marginal is ~0 and non-monotonic near the
        root. Ranking the beam by that marginal makes the search abandon the true
        path in the initial dip. So we **drive the beam with the question-priming
        differential** ``target - base`` (the side channel proper -- it robustly
        surfaces the next reasoning token), exactly like ``continuation``, and use
        the public answer A only as an **anchor**: among the primed candidates we
        score cumulative ``log P(A | Q, <think> prefix </think>)`` and keep, as
        the extraction, the prefix that maximises it -- provided that maximum
        clears ``gain_threshold`` over the empty-reasoning control. If the answer
        likelihood never meaningfully rises (no channel), we extract nothing.

        ``delta_threshold`` bounds the reasoning (a natural end when the priming
        signal fades); ``gain_threshold`` is the absolute answer-likelihood gain
        the winning prefix must clear to be emitted.
        """
        base_ll = self.oracle.base_answer_loglik()
        beams = [Beam(neg_score=0.0, seq=(), adelta=0.0)]  # neg_score = -cumulative priming
        best_seq: Tuple[int, ...] = ()
        best_adelta = 0.0
        signal: List[float] = []

        for _ in range(self.max_tokens):
            if all(b.finished for b in beams):
                break
            candidates: List[Beam] = []
            step_best_gain = float("-inf")

            for b in beams:
                if b.finished:
                    candidates.append(b)
                    continue
                # Proposal + search driver: question-priming delta (target-base).
                proposals = [(d, t) for d, t in self._propose(b.seq)
                             if t != self.oracle.think_close_id]
                kept = [(d, t) for d, t in proposals if d >= self.delta_threshold]
                if not kept:
                    b.finished = True  # priming faded -> reasoning has ended
                    candidates.append(b)
                    continue

                # Anchor: one batched teacher-forced forward scores log P(A) for
                # every primed candidate; the answer decides the truncation point.
                child_lls = self.oracle.answer_loglik_batch([b.seq + (t,) for _, t in kept])
                parent_prime = -b.neg_score
                for (d, t), child_ll in zip(kept, child_lls):
                    adelta = child_ll - base_ll
                    step_best_gain = max(step_best_gain, adelta - b.adelta)
                    candidates.append(
                        Beam(neg_score=-(parent_prime + d), seq=b.seq + (t,), adelta=adelta)
                    )
                    if adelta > best_adelta and adelta >= self.gain_threshold:
                        best_adelta, best_seq = adelta, b.seq + (t,)

            signal.append(step_best_gain if step_best_gain != float("-inf") else 0.0)
            beams = heapq.nsmallest(self.beam_width, candidates)  # rank by cumulative priming

        return [Beam(neg_score=-best_adelta, seq=best_seq)], signal

    # -- continuation search (cheap variant) ------------------------------- #
    def _run_continuation(self):
        beams = [Beam(neg_score=0.0, seq=(), finished=False)]
        signal: List[float] = []

        for _ in range(self.max_tokens):
            if all(b.finished for b in beams):
                break
            candidates: List[Beam] = []
            step_max = 0.0
            for b in beams:
                if b.finished:
                    candidates.append(b)
                    continue
                proposals = self._propose(b.seq)
                best_delta = proposals[0][0] if proposals else 0.0
                step_max = max(step_max, best_delta)
                if best_delta < self.delta_threshold:
                    b.finished = True
                    b.dead = True
                    candidates.append(b)
                    continue
                for delta_val, t in proposals:
                    if delta_val < self.delta_threshold:
                        break
                    seq = b.seq + (t,)
                    mean = (-b.neg_score * len(b.seq) + delta_val) / len(seq)
                    finished = t == self.oracle.think_close_id
                    candidates.append(Beam(neg_score=-mean, seq=seq, finished=finished))
            signal.append(step_max)
            beams = heapq.nsmallest(self.beam_width, candidates)

        return beams, signal


def _counter_value(oracle) -> int:
    counter = getattr(oracle, "counter", None)
    return counter.forward_passes if counter is not None else 0


# --------------------------------------------------------------------------- #
# Convenience builder for the CoT threat model
# --------------------------------------------------------------------------- #
def build_cot_oracle(model, tokenizer, question: str, answer_text: str,
                     counter: Optional[QueryCounter] = None):
    """Construct an HFOracle for the (question, public answer) the attacker holds.

    target context = chat_template(question) + "<think>\\n"  (question-primed)
    base context   = "<think>\\n"                            (question removed)
    answer         = tokenised public answer A (the extraction anchor)
    """
    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Solve the following problem."},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    open_think = "<think>\n" if "<think>" not in prompt else ""
    context_ids = tokenizer.encode(prompt + open_think, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    return HFOracle(model, tokenizer, context_ids, answer_ids, counter=counter)
