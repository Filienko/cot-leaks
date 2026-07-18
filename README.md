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

## Method

For a reconstructed prefix `c_{<t}`:

```
delta = logits_target(context + c_{<t})  -  logits_base(c_{<t})
```

- **target** — the instance primed with the hidden context (the victim's
  question sitting in front of its hidden `<think>` trace).
- **base** — a zero-context control: the same model and prefix with the priming
  context removed, giving the pure LM prior.

`delta ~ 0` ⇒ ordinary LM continuation → **pruned**. `delta` above a threshold
⇒ **context-primed** → beam candidate. An autoregressive **backtracking beam
search** appends the highest-delta token, advances, and backtracks when a path
goes flat (no spike = wrong turn). The search runs entirely over **token ids**
to avoid BPE round-trip corruption (`decode`→`encode` is not identity).

## Files

| file | role |
|------|------|
| `reconstruction_poc.py` | entrypoint: ground-truth capture, baselines, attack, report |
| `attack.py` | differential `target − base` beam search; `LogitOracle` / `HFOracle` |
| `baselines.py` | text-layer baselines: naive direct prompt + jailbreak |
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

## Threat model

The attacker observes the victim's question and public answer, and can query the
target model's next-token logits (open-weights: full vocab; or an API exposing
logprobs / token biases). They also control a base instance of the same model
with no hidden context. The default target is
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (open weights, full logit access).

## Running

```bash
pip install -r requirements.txt          # needs a GPU for the default 8B model

python reconstruction_poc.py \
    --model-id deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --beam-width 4 --top-k 8 --delta-threshold 1.0 --max-tokens 64

python reconstruction_poc.py --use-gsm8k   # first GSM8K test question
python reconstruction_poc.py --no-embedding  # skip the embedding metric
```

`--delta-threshold` is the key knob: raise it to demand stronger priming
evidence per token (higher precision, more backtracking); lower it to extract
more aggressively.

### Self-tests (no GPU)

```bash
python tests/test_mock.py
```

Verifies the beam mechanics (exact recovery of a planted hidden sequence,
backtracking on a flat signal) and the metric battery with a numpy mock oracle —
no model download required.
