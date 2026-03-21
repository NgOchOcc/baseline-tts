import os
import json
import sys
from typing import List, Optional, Any
from tqdm import tqdm

# Add MATH folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/envs/MATH'))

from grader import math_equal
from verify_utils import grade_answer
from parse_utils_qwen import extract_answer


def is_correct_with_math_verify(predicted: Any, groundtruth: Any) -> bool:
    """
    Check correctness using MATH folder evaluation approach.
    Tries both grader.math_equal and verify_utils.grade_answer for robustness.
    """
    try:
        pred_str = str(predicted).strip()
        gt_str = str(groundtruth).strip()

        if not pred_str or not gt_str:
            return False

        # Try math_equal from grader.py first (most comprehensive)
        try:
            if math_equal(pred_str, gt_str, include_percentage=True, is_close=True):
                return True
        except Exception:
            pass

        # Fallback to grade_answer from verify_utils.py
        try:
            if grade_answer(pred_str, gt_str):
                return True
        except Exception:
            pass

        return False
    except Exception:
        return False


def pass_at_k(
    extracted_answers: List[Optional[str]],
    groundtruth: str,
    k: int,
) -> bool:
    """Return True if any of top-k answers is correct."""
    valid_answers = [
        a for a in extracted_answers
        if a is not None and str(a).strip() != ""
    ]

    if not valid_answers:
        return False

    top_k_answers = valid_answers[:min(k, len(valid_answers))]

    for ans in top_k_answers:
        if is_correct_with_math_verify(ans, groundtruth):
            return True

    return False


# ================== CONFIG ==================
seeds = range(1)

ks = [1, 8, 16]

data_name = "math"  # Dataset name for answer extraction

base_root = (
    "/Users/luungoc/Project/compute-optimal-tts/Qwen2.5-7B-Instruct/Qwen2.5-Math-PRM-7B"
)
# ===========================================


for seed in seeds:
    base_dir = f"{base_root}/seed_0_width_16_num_seq_16_num_q_0"
    print(f"Processing: {base_dir}")

    pass_counts = {k: 0 for k in ks}
    total = 0

    # Get all question directories for progress tracking
    question_dirs = [d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d))]

    for question_dir in tqdm(question_dirs, desc="Processing questions", unit="q"):
        question_path = os.path.join(base_dir, question_dir)

        extracted_answers: List[Optional[str]] = []
        groundtruth: Optional[str] = None

        # ── collect all candidate answers ──────────────────────────
        jsonl_files = [f for f in os.listdir(question_path)
                       if f.endswith(".jsonl")]

        for filename in tqdm(jsonl_files, desc=f"  {question_dir}",
                             leave=False, unit="file"):
            file_path = os.path.join(question_path, filename)

            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)

                    groundtruth = data.get("groundtruth", None)

                    for out in data.get("output", []):
                        ans = out.get("extracted_answer", None)
                        extracted_answers.append(ans)

        # ── compute pass@k ─────────────────────────────────────────
        if groundtruth is not None:
            # Extract answer from ground truth
            gt_answer = extract_answer(groundtruth, data_name)

            if gt_answer:  # Only evaluate if ground truth answer was extracted
                total += 1
                for k in tqdm(ks, desc=f"  Computing pass@k",
                             leave=False, unit="k"):
                    if pass_at_k(extracted_answers, gt_answer, k):
                        pass_counts[k] += 1

    # ── print results ─────────────────────────────────────────────
    print(f"Seed {seed}")
    print(f"Total questions : {total}")

    for k in ks:
        acc = pass_counts[k] / total if total > 0 else 0.0
        print(f"pass@{k:<2}        : {acc:.4f}")

    print("-" * 50)