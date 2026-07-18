# cot-leaks

Reconstructing a reasoning model's **hidden chain-of-thought** through a
**differential logit side channel** — a structural, architecture-level attack
rather than a prompt-layer one.

## Thesis

Text-layer defenses (refusal training, output filtering) sit *above* the logit
layer. Holding a hidden context in the window still leaves a measurable,
anomalous **upward shift** in the logits of the tokens that make up that
context, beyond background language-modelling noise. We isolate that shift with
a differential query and use it to reconstruct the hidden reasoning token by
token.

## Threat model

The attacker knows the user's **question Q** and the model's **public answer A**
(the output *not* between the `<think>` tags). The attacker does **not** know the
hidden `<think>` reasoning — that is the extraction target. The attacker can
query the model's next-token logits (open weights: full vocab; or an API exposing
logprobs). Default target: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`.

## Method

### `answer_anchored` (default — the real inversion)

The public answer **A is the anchor**. We reconstruct the hidden CoT `c` token by
token to maximise how well the model explains the *known* answer, with an
**empty-reasoning control** subtracted:

```
score(c) = log P(A | Q, <think> c </think>)  −  log P(A | Q, <think></think>)
                    └── candidate reasoning ──┘        └──── empty control ────┘
```

The control is the "zero-context" query: the answer's likelihood with **no
reasoning at all**. Subtracting it isolates what the reasoning *contributes* to
the answer, so the score rewards tokens that explain A rather than tokens that
are merely fluent. A token is accepted only if it raises `score` by at least
`--gain-threshold` **nats**; otherwise the path has gone flat (wrong turn /
natural end) and the **backtracking beam search** drops it. Candidate tokens are
proposed by the question-priming delta `logits_target − logits_base` so only
context-relevant tokens are ever scored. Everything runs over **token ids** to
avoid BPE round-trip corruption (`decode`→`encode` is not identity).

This uses exactly what the attacker holds (Q and A) and never reads the `<think>`
tokens. It genuinely *inverts*: given Q and A, recover the reasoning that bridges
them.

### `priming` (`--method priming`) — the brief-faithful pure side channel

This is the differential the project brief describes literally, and it needs
**no answer A**. A *target* instance **holds the hidden CoT in its context**; a
*zero-context control* instance does not. For candidate prefix `c`:

```
target = model( Q <think> C </think> [A] + "repeat your reasoning…<think>" + c )
base   = model( Q                        + "repeat your reasoning…<think>" + c )
delta  = logits(target) − logits(base)    →  the true next token of C spikes
```

The in-context `C` primes its own tokens (verbatim-copy / induction behaviour),
so the true next token shows an anomalous positive delta while LM artifacts
cancel. Walk `c` forward, accepting tokens with `delta > --delta-threshold`,
backtracking when the spike vanishes. The control is **base = the query Q** (not
empty): because the attacker knows Q, Q cancels in the delta and only the unknown
`C` is isolated. The extractor **only reads logits** — it never reads `C`; the
elicitation string merely puts the target into a re-emit state.

Realism caveat: this assumes you can query a target instance that *still holds*
the hidden CoT and read its logits. Deployed reasoning APIs typically discard the
hidden CoT before the next turn and never expose such logits, so `priming` is
honestly an **open-weights PoC** — the harness stands up the victim session, but
the attack path only appends the elicitation + its reconstruction and reads
logits. For it to demonstrate anything beyond regeneration, the victim CoT must
be **sampled** (`--sample-cot`, temp>0, seed the attacker lacks) or shaped by a
hidden system prompt; otherwise "target holds C" collapses into "just re-run Q".
The `greedy regeneration` baseline in the report is the yardstick: `priming`
should recover the victim's *specific* sampled trace with higher verbatim /
embedding similarity than blind regeneration.

### `continuation` (cheap variant, `--method continuation`)

`delta = logits_target(Q + <think> + c) − logits_base(<think> + c)` isolates the
question's priming of the next reasoning token; a token is kept if
`delta > --delta-threshold` logits. Fast (2 forwards/step) but, since the
attacker already holds Q, this *regenerates* the trace rather than inverting
against A — it's the proposal signal, exposed on its own for comparison.

### Why not just match logits at one answer position (the original PoC)

The first-cut PoC scored candidates by the L2 distance between logits at a single
position right after a fixed answer prefix. That position's next token is
dominated by the answer prefix, not the reasoning, so the signal is degenerate —
every candidate looks alike. Combined with a cumulative loss that only ever grows
per step (while a finished `</think>` path freezes), the search collapses to
emitting `</think>` immediately or locking in noise. The answer-anchored score
fixes both: it scores the likelihood of the **whole** answer (sensitive to `c`)
with a **standalone** per-token gain (no length bias).

## Files

| file | role |
|------|------|
| `reconstruction_poc.py` | entrypoint: ground-truth capture, baselines, attack, report |
| `attack.py` | differential `target − base` beam search; `LogitOracle` / `HFOracle` |
| `baselines.py` | baselines: naive prompt, jailbreak, greedy regeneration |
| `metrics.py` | fidelity metrics + query accounting |
| `tests/test_mock.py` | torch-free self-tests (numpy mock oracle) |

## Metrics

We report a battery, because no single number captures extraction fidelity:

- **Exact / verbatim overlap** — the strict string-matching view used by
  Carlini et al. (training-data extraction, USENIX Sec 2021; memorization
  quantification, 2023): contiguous verbatim reproduction + Carlini-style
  n-gram extraction rate. Conservative lower bound.
- **Token edit distance (Levenshtein), BLEU, ROUGE-L** — approximate string
  matching.
- **Embedding cosine similarity** — semantic view. Per Google,
  [arXiv:2510.18554](https://arxiv.org/pdf/2510.18554), string matching
  *severely undercounts* extraction (their estimate: by a large factor) because
  trivial surface artifacts deflate edit-distance-style metrics, while a
  high-quality embedding model recovers the semantic overlap they miss.
- **Queries per token** — forward passes / tokens extracted (efficiency).

Optional metric deps (`sacrebleu`, `rouge_score`, `sentence-transformers`)
degrade to `n/a` in the report if not installed; the exact/edit/F1 metrics
always run.

## Running

```bash
pip install -r requirements.txt          # needs a GPU for the default 8B model

# Default: answer-anchored inversion against the public answer A.
python reconstruction_poc.py \
    --model-id deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --beam-width 4 --top-k 8 --gain-threshold 0.05 --max-tokens 64

# Brief-faithful pure side channel against a *sampled* victim CoT (no A needed).
python reconstruction_poc.py --method priming --sample-cot --temperature 0.7 \
    --delta-threshold 1.0 --max-tokens 64

python reconstruction_poc.py --method continuation --delta-threshold 1.0
# Question source: defaults to GSM8K "socratic" (test split, row 0).
python reconstruction_poc.py --gsm8k-index 3           # a different socratic row
python reconstruction_poc.py --gsm8k-config main       # the plain GSM8K subset
python reconstruction_poc.py --gsm8k-config ""         # built-in DEFAULT_QUESTION
python reconstruction_poc.py --question "..."          # your own question
python reconstruction_poc.py --no-embedding  # skip the embedding metric
```

Key knobs:

- `--gain-threshold` (answer-anchored): minimum gain in `log P(A)` (nats) to
  accept a token. Model/dataset dependent — raise for precision + more
  backtracking, lower to extract more aggressively. Start at `0.05` and tune.
- `--delta-threshold` (continuation / priming): minimum priming logit delta to
  accept a token.
- `--sample-cot` / `--temperature` / `--victim-seed`: make the victim's hidden
  CoT non-trivially regenerable (the meaningful setup for `priming`).

The report prints **full, untruncated** outputs for every method (ground-truth
CoT, public answer, all three baselines, and the reconstruction) plus the metrics
table and query cost.

### Self-tests (no GPU)

```bash
python tests/test_mock.py
```

Verifies the beam mechanics for all three methods (exact recovery of a planted
hidden sequence, backtracking on a flat signal) and the metric battery with a
numpy mock oracle — no model download required.
