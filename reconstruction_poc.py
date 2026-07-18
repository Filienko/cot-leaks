"""
Proof-of-concept: reconstructing a reasoning model's hidden chain-of-thought
through a differential logit side channel, benchmarked against text-layer
baselines.

Pipeline
--------
1. Generate ground truth: run the target model on a math question and capture
   its hidden <think> reasoning + public answer (the answer is the only thing a
   real victim would expose).
2. Baselines: naive "repeat your reasoning" prompt, a jailbreak prompt, and a
   greedy regeneration (attacker's guess without the channel) (baselines.py).
3. Side channel: the differential (target - base) beam search over token ids
   (attack.py) -- answer_anchored (default), continuation, or priming (target
   holds the hidden CoT, base = query only; the brief-faithful side channel).
4. Evaluation: score every method against the true hidden CoT with the metric
   battery in metrics.py (Carlini-style exact overlap, edit distance, BLEU,
   ROUGE, and embedding similarity), and report queries-per-token.

Requires a GPU for the default DeepSeek-R1-Distill model. See README.md.
"""

from __future__ import annotations

import argparse

import metrics
from attack import DifferentialExtractor, build_cot_oracle, build_priming_oracle
from baselines import (
    extract_reasoning_copy,
    run_greedy_regeneration,
    run_jailbreak_baseline,
    run_naive_baseline,
)


DEFAULT_QUESTION = (
    "A maker of 3D printers buys plastic in spools of 5 kilograms. Each printer "
    "requires 2.5 kilograms of plastic. If the maker buys 10 spools, how many "
    "printers can they make?"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--question", default=None,
                   help="Override the math question. Takes precedence over the GSM8K loader.")
    p.add_argument("--gsm8k-config", default="socratic",
                   help="GSM8K subset to draw the question from (e.g. 'socratic' or 'main'). "
                        "Set to '' to skip GSM8K and use the built-in DEFAULT_QUESTION.")
    p.add_argument("--gsm8k-split", default="test",
                   help="GSM8K split to draw the question from.")
    p.add_argument("--gsm8k-index", type=int, default=0,
                   help="Row index of the GSM8K question to use.")
    p.add_argument("--method", choices=["answer_anchored", "continuation", "priming"],
                   default="answer_anchored",
                   help="answer_anchored: invert against the public answer A (default). "
                        "continuation: cheap question-priming regeneration. "
                        "priming: brief-faithful side channel -- target holds the hidden "
                        "CoT, base = query only, reconstruct from target-base delta (no A).")
    p.add_argument("--sample-cot", action="store_true",
                   help="Sample the victim's hidden CoT (temp>0) instead of greedy, so it is "
                        "not trivially regenerable -- the meaningful setup for --method priming.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature for the victim CoT when --sample-cot is set.")
    p.add_argument("--victim-seed", type=int, default=None,
                   help="Seed for the victim's sampled CoT. The attacker does NOT know it; "
                        "left unset it is random each run.")
    p.add_argument("--beam-width", type=int, default=4)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--delta-threshold", type=float, default=1.0,
                   help="continuation only: min target-base logit delta to accept a token.")
    p.add_argument("--gain-threshold", type=float, default=0.05,
                   help="answer_anchored only: min cumulative gain (nats) in log P(A) over the "
                        "empty-reasoning control that the emitted prefix must clear. Model/dataset "
                        "dependent -- raise for precision, lower to extract more.")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="Max CoT tokens to reconstruct via the side channel.")
    p.add_argument("--gt-max-new-tokens", type=int, default=400,
                   help="Max tokens when generating the ground-truth trace.")
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                   help="sentence-transformers model for semantic similarity (arXiv:2510.18554).")
    p.add_argument("--no-embedding", action="store_true", help="Skip embedding similarity.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # transformers renamed the precision kwarg torch_dtype -> dtype; older
    # releases only accept torch_dtype, so try the new name and fall back.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", dtype=torch.bfloat16
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
    model.eval()
    return model, tokenizer


def get_question(args) -> str:
    if args.question:
        return args.question
    if args.gsm8k_config:
        try:
            from datasets import load_dataset

            ds = load_dataset("openai/gsm8k", args.gsm8k_config, split=args.gsm8k_split)
            question = ds[args.gsm8k_index]["question"]
            print(f"Using GSM8K/{args.gsm8k_config}[{args.gsm8k_split}][{args.gsm8k_index}].")
            return question
        except Exception as e:
            print(f"GSM8K load failed ({e}); using the built-in question.")
    return DEFAULT_QUESTION


def generate_ground_truth(model, tokenizer, question: str, max_new_tokens: int,
                          do_sample: bool = False, temperature: float = 0.7,
                          seed: int = None):
    """Return (cot_text, cot_ids, answer_text) for the target's hidden trace.

    We split on the ``</think>`` *token id* rather than the decoded string and
    keep special tokens, so template-opened ``<think>`` and special-token
    encodings of the tags don't break the parse (the original PoC's bug).
    """
    import torch

    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Solve the following problem."},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    mode = f"sampled (temp={temperature}, seed={seed})" if do_sample else "greedy"
    print(f"\n[Target] Generating ground-truth reasoning + answer ({mode}) ...")
    if do_sample and seed is not None:
        torch.manual_seed(seed)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.no_grad():
        gen = model.generate(**inputs, **gen_kwargs)
    gen_ids = gen[0][inputs.input_ids.shape[1]:].tolist()

    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    close_id = close_ids[-1] if close_ids else None
    open_ids = set(tokenizer.encode("<think>", add_special_tokens=False))

    if close_id is not None and close_id in gen_ids:
        cut = gen_ids.index(close_id)
        cot_ids = [t for t in gen_ids[:cut] if t not in open_ids]
        answer_ids = gen_ids[cut + 1:]
    else:
        # No explicit closing tag: treat the leading span as reasoning.
        cot_ids = [t for t in gen_ids if t not in open_ids]
        answer_ids = []

    cot_text = tokenizer.decode(cot_ids, skip_special_tokens=True).strip()
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return cot_text, cot_ids, answer_text


def main():
    args = parse_args()

    try:
        import torch

        torch.manual_seed(args.seed)
    except ImportError:
        pass

    model, tokenizer = load_model(args.model_id)
    tokenize = lambda s: tokenizer.encode(s, add_special_tokens=False)

    embedding_scorer = (
        None if args.no_embedding else metrics.EmbeddingScorer(args.embedding_model)
    )

    question = get_question(args)
    cot_text, cot_ids, answer_text = generate_ground_truth(
        model, tokenizer, question, args.gt_max_new_tokens,
        do_sample=args.sample_cot, temperature=args.temperature, seed=args.victim_seed,
    )

    # --- Text-layer baselines -------------------------------------------- #
    naive_counter = metrics.QueryCounter()
    jb_counter = metrics.QueryCounter()
    regen_counter = metrics.QueryCounter()
    print("\n[Baseline] Naive direct prompt ...")
    naive_raw = run_naive_baseline(model, tokenizer, question, counter=naive_counter)
    print("[Baseline] Jailbreak prompt ...")
    jb_raw = run_jailbreak_baseline(model, tokenizer, question, counter=jb_counter)
    print("[Baseline] Greedy regeneration (attacker's guess without the channel) ...")
    regen_pred = run_greedy_regeneration(model, tokenizer, question, counter=regen_counter)
    naive_pred = extract_reasoning_copy(naive_raw)
    jb_pred = extract_reasoning_copy(jb_raw)

    # --- Differential side channel --------------------------------------- #
    print(f"\n[Side channel] Differential attack (method={args.method}) ...")
    sc_counter = metrics.QueryCounter()
    if args.method == "priming":
        # Target instance holds the hidden CoT; the extractor never reads it.
        oracle = build_priming_oracle(
            model, tokenizer, question, cot_ids, answer_text=answer_text,
            counter=sc_counter,
        )
    else:
        oracle = build_cot_oracle(model, tokenizer, question, answer_text, counter=sc_counter)
    extractor = DifferentialExtractor(
        oracle,
        method=args.method,
        beam_width=args.beam_width,
        top_k=args.top_k,
        delta_threshold=args.delta_threshold,
        gain_threshold=args.gain_threshold,
        max_tokens=args.max_tokens,
    )
    result = extractor.run(decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True))

    # --- Evaluation ------------------------------------------------------- #
    def score(pred_text):
        return metrics.evaluate(pred_text, cot_text, tokenize=tokenize,
                                embedding_scorer=embedding_scorer)

    naive_scores = score(naive_pred)
    jb_scores = score(jb_pred)
    regen_scores = score(regen_pred)
    sc_scores = score(result.text)

    victim_mode = (f"sampled (temp={args.temperature}, seed={args.victim_seed})"
                   if args.sample_cot else "greedy")

    print("\n" + "=" * 96)
    print("HIDDEN-CoT EXTRACTION REPORT")
    print("=" * 96)
    print(f"\n[Question]\n{question}")
    print(f"\n[Ground-truth hidden CoT -- victim {victim_mode}]  ({len(cot_ids)} tokens)\n{cot_text}")
    print(f"\n[Public answer A -- known to the attacker]\n{answer_text}")

    # Full, untruncated raw outputs for each method.
    print("\n" + "-" * 96)
    print("[Baseline: naive direct prompt] -- full raw output")
    print("-" * 96)
    print(naive_raw)
    print("\n" + "-" * 96)
    print("[Baseline: jailbreak prompt] -- full raw output")
    print("-" * 96)
    print(jb_raw)
    print("\n" + "-" * 96)
    print("[Baseline: greedy regeneration] -- full reconstructed CoT (no side channel)")
    print("-" * 96)
    print(regen_pred)
    print("\n" + "-" * 96)
    print(f"[Side channel: {args.method}] -- full reconstructed hidden CoT "
          f"({len(result.token_ids)} tokens)")
    print("-" * 96)
    print(result.text)

    # Metrics table (vs. the true hidden CoT).
    print("\n" + "-" * 96)
    print(f"{'method':<26}{'queries':>9}  metrics (vs. true hidden CoT)")
    print("-" * 96)
    for name, scores, q in (
        ("naive prompt", naive_scores, naive_counter.forward_passes),
        ("jailbreak prompt", jb_scores, jb_counter.forward_passes),
        ("greedy regeneration", regen_scores, regen_counter.forward_passes),
        (f"side channel [{args.method}]", sc_scores, result.forward_passes),
    ):
        print(f"{name:<26}{q:>9}  {scores.as_row()}")

    print("-" * 96)
    print(
        f"side-channel queries/token: {result.queries_per_token():.2f}  "
        f"({result.forward_passes} passes / {len(result.token_ids)} tokens, "
        f"{result.steps} steps)"
    )
    if result.method == "answer_anchored":
        print(f"final delta log P(A) (answer explained by reconstructed CoT): {result.score:.3f} nats")
    if result.per_position_signal:
        avg = sum(result.per_position_signal) / len(result.per_position_signal)
        label = ("mean per-token answer-likelihood gain (nats)"
                 if result.method == "answer_anchored"
                 else "mean per-position max priming delta (logits)")
        print(f"{label}: {avg:.3f}")
    print("=" * 96)


if __name__ == "__main__":
    main()
