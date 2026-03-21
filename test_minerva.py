"""
eval_cot_baseline.py
────────────────────
Plain chain-of-thought baseline evaluation using vLLM + math_verify.

Usage:
    python eval_cot_baseline.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset data/math500.json \
        --dataset_type math500 \
        --output results/cot_baseline.jsonl \
        --num_samples 100 \
        --n 1 \
        --temperature 0.0 \
        --max_tokens 2048

n > 1  → majority vote (pass@1 via majority) + best-of-n oracle
n = 1  → greedy / single sample
"""

import os
import re
import json
import argparse
import logging
from collections import Counter
from typing import List, Optional, Tuple

from tqdm import tqdm

# ── vLLM ──────────────────────────────────────────────────────────────────────
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
from vllm import LLM, SamplingParams

from math_verify import verify, parse
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Answer utilities
# ══════════════════════════════════════════════════════════════════════════════

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} content with proper nested brace handling."""
    results = []
    idx = 0
    while True:
        start = text.find(r'\boxed{', idx)
        if start == -1:
            break
        brace_start = start + len(r'\boxed{')
        depth = 1
        i = brace_start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[brace_start:i-1].strip())
        idx = i
    return results[-1] if results else None


def verify_answer(
    response: str,
    ground_truth: str,
    use_math_verify: bool = True,
) -> bool:
    """Verify response against ground truth using math_verify."""
    if use_math_verify:
        gold_parsed = parse(
            f"\\boxed{{{ground_truth}}}",
            extraction_config=[LatexExtractionConfig()]
        )
        pred_parsed = parse(
            response,
            extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()]
        )
        return bool(verify(gold_parsed, pred_parsed))
    # fallback: extract boxed + string match
    pred = extract_boxed_answer(response)
    if pred is None:
        return False
    return pred.strip() == ground_truth.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loading  (mirrors tree/evaluate.py)
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_sample(s: dict) -> dict:
    """
    Normalize a raw sample dict so it always has keys 'problem' and 'answer'.
    Handles datasets that use 'question' instead of 'problem',
    and 'solution' / 'label' instead of 'answer'.
    """
    # ── problem key ──────────────────────────────────────────────────────────
    if "problem" not in s:
        for k in ("question", "query", "input", "prompt"):
            if k in s:
                s["problem"] = s[k]
                break

    # ── answer key ───────────────────────────────────────────────────────────
    if "answer" not in s:
        for k in ("solution", "label", "target", "output", "ground_truth"):
            if k in s:
                s["answer"] = str(s[k])
                break

    return s


def load_dataset(dataset_path: str, dataset_type: str = "math500") -> List[dict]:
    if dataset_type in ("math500", "gsm8k", "math500_ablation"):
        if dataset_path.endswith(".jsonl"):
            with open(dataset_path) as f:
                raw = [json.loads(l) for l in f if l.strip()]
        else:
            with open(dataset_path) as f:
                raw = json.load(f)
        # auto-normalize keys (handles 'question' → 'problem' etc.)
        return [_normalize_sample(s) for s in raw]

    elif dataset_type in ("aime", "aime24", "amc"):
        samples = []
        with open(dataset_path) as f:
            for line in f:
                if not line.strip():
                    continue
                s = json.loads(line)
                samples.append({
                    "problem": s["problem"],
                    "answer":  str(s["extracted_groundtruth"]),
                    "solution": s.get("solution", ""),
                    "url":      s.get("url", ""),
                })
        return samples

    elif dataset_type in ("aime25", "aime26"):
        samples = []
        with open(dataset_path) as f:
            for line in f:
                if not line.strip():
                    continue
                s = json.loads(line)
                samples.append({
                    "problem": s["problem"],
                    "answer":  str(s["answer"]),
                    "id":      s.get("id"),
                })
        return samples

    elif dataset_type == "minerva":
        samples = []
        with open(dataset_path) as f:
            for line in f:
                if not line.strip():
                    continue
                s = json.loads(line)
                samples.append({
                    "problem": s["question"],
                    "answer":  str(s["answer"]),
                })
        return samples

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


# ══════════════════════════════════════════════════════════════════════════════
# Prompt formatting
# ══════════════════════════════════════════════════════════════════════════════

INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

def format_prompt(problem: str, tokenizer, is_math_model: bool = False) -> str:
    """Format a single problem into a chat prompt."""
    if hasattr(tokenizer, "apply_chat_template"):
        if is_math_model:
            messages = [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user",   "content": problem},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{problem}\n{INSTRUCTION}"},
            ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return f"{problem}\n{INSTRUCTION}"


# ══════════════════════════════════════════════════════════════════════════════
# Answer aggregation helpers
# ══════════════════════════════════════════════════════════════════════════════

def majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    """Return the most common non-None answer (ties → first)."""
    valid = [a for a in answers if a]
    if not valid:
        return None
    counts = Counter(valid)
    return counts.most_common(1)[0][0]


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    # ── load dataset ──────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset}  type={args.dataset_type}")
    dataset = load_dataset(args.dataset, args.dataset_type)
    if args.num_samples is not None:
        dataset = dataset[: min(args.num_samples, len(dataset))]
    logger.info(f"  → {len(dataset)} problems")

    # ── load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    is_math_model = "math" in args.model.lower()
    logger.info(f"  → is_math_model={is_math_model}")

    # ── sampling params ────────────────────────────────────────────────────────
    params = SamplingParams(
        n=args.n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # ── output file ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # ── build all prompts upfront ─────────────────────────────────────────────
    logger.info("Formatting prompts...")
    prompts = [format_prompt(s["problem"], tokenizer, is_math_model) for s in dataset]

    # ── single batched generate call ─────────────────────────────────────────
    logger.info(f"Generating {len(prompts)} problems in one batch  "
                f"(n={args.n}, temp={args.temperature})...")
    all_raw = llm.generate(prompts, sampling_params=params, use_tqdm=True)
    logger.info("Generation done. Running eval...")

    # ── eval loop (CPU only, fast) ────────────────────────────────────────────
    correct_mv     = 0
    correct_oracle = 0
    total          = 0
    all_tokens     = []
    pass_at_1_list = []

    with open(args.output, "w") as out_f:
        for idx, (sample, raw) in enumerate(zip(dataset, all_raw)):
            problem      = sample["problem"]
            ground_truth = sample["answer"]
            outputs      = raw.outputs  # list of n CompletionOutput

            responses        = [o.text for o in outputs]
            total_new_tokens = sum(len(o.token_ids) for o in outputs)
            all_tokens.append(total_new_tokens)

            pred_answers = [extract_boxed_answer(r) for r in responses]

            correct_flags = [
                verify_answer(r, ground_truth, args.use_math_verify)
                for r in responses
            ]

            # majority vote
            mv_answer   = majority_vote(pred_answers)
            mv_response = responses[0]
            if mv_answer is not None:
                for r, a in zip(responses, pred_answers):
                    if a == mv_answer:
                        mv_response = r
                        break

            mv_correct     = verify_answer(mv_response, ground_truth, args.use_math_verify)
            oracle_correct = any(correct_flags)

            if mv_correct:     correct_mv += 1
            if oracle_correct: correct_oracle += 1
            total += 1

            frac_correct = sum(correct_flags) / len(correct_flags)
            pass_at_1_list.append(frac_correct)

            record = {
                "idx":              idx,
                "problem":          problem,
                "ground_truth":     ground_truth,
                "mv_response":      mv_response,
                "mv_answer":        mv_answer if mv_answer else "",
                "mv_correct":       mv_correct,
                "oracle_correct":   oracle_correct,
                "all_responses":    responses,
                "all_answers":      [a if a else "" for a in pred_answers],
                "all_correct":      correct_flags,
                "pass_at_1_frac":   round(frac_correct, 4),
                "total_new_tokens": total_new_tokens,
            }
            if "url" in sample:
                record["url"] = sample["url"]

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            if (idx + 1) % 10 == 0 or (idx + 1) == len(dataset):
                acc_mv     = correct_mv     / total * 100
                acc_oracle = correct_oracle / total * 100
                avg_tok    = sum(all_tokens) / len(all_tokens)
                logger.info(
                    f"[{idx+1}/{len(dataset)}]  "
                    f"MV acc={acc_mv:.2f}%  oracle acc={acc_oracle:.2f}%  "
                    f"avg_tokens={avg_tok:.0f}"
                )

    # ── final summary ──────────────────────────────────────────────────────────
    acc_mv     = correct_mv     / total * 100
    acc_oracle = correct_oracle / total * 100
    avg_tokens = sum(all_tokens) / len(all_tokens) if all_tokens else 0
    mean_pass1 = sum(pass_at_1_list) / len(pass_at_1_list) * 100 if pass_at_1_list else 0

    print("\n" + "=" * 60)
    print("CoT Baseline Evaluation Results")
    print("=" * 60)
    print(f"Model            : {args.model}")
    print(f"Dataset          : {args.dataset}  ({args.dataset_type})")
    print(f"Problems         : {total}")
    print(f"n (samples/prob) : {args.n}")
    print(f"Temperature      : {args.temperature}")
    print(f"Max tokens       : {args.max_tokens}")
    print("-" * 60)
    print(f"Accuracy (majority vote)          : {acc_mv:.2f}%  ({correct_mv}/{total})")
    print(f"Accuracy (oracle / best-of-{args.n})  : {acc_oracle:.2f}%  ({correct_oracle}/{total})")
    print(f"Mean pass@1 (fraction correct)    : {mean_pass1:.2f}%")
    print(f"Avg tokens per problem            : {avg_tokens:.0f}")
    print("=" * 60)
    print(f"Results saved to : {args.output}")

    return acc_mv


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="CoT Baseline Evaluation with vLLM")

    # model
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=4096)

    # dataset
    p.add_argument("--dataset", default="data/math500.json")
    p.add_argument(
        "--dataset_type",
        default="math500",
        choices=["math500", "aime", "aime24", "aime25", "aime26", "amc", "gsm8k", "minerva", "math500_ablation"],
    )
    p.add_argument("--num_samples", type=int, default=None,
                   help="Limit number of problems (default: all)")

    # sampling
    p.add_argument("--n", type=int, default=1,
                   help="Number of completions per problem (1=greedy baseline, >1=best-of-n/MV)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 for greedy, >0 for sampling")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=2048)

    # eval
    p.add_argument("--use_math_verify", action="store_true", default=True)
    p.add_argument("--no_math_verify", dest="use_math_verify", action="store_false")

    # output
    p.add_argument("--output", default="results/cot_baseline.jsonl")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)