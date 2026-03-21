import os
import json
import sys
from collections import Counter
from typing import List, Optional, Any, Tuple
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import lru_cache

# Add MATH folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/envs/MATH'))

from grader import math_equal
from verify_utils import grade_answer
from parse_utils_qwen import extract_answer


def get_value_from_path(value_list: List[float], mode: str = "avg") -> float:
    """Aggregate a per-step PRM value list into a single scalar."""
    if not value_list:
        return 0.0
    if mode == "max":
        return max(value_list)
    elif mode == "last":
        return value_list[-1]
    else:  # "avg"
        return sum(value_list) / len(value_list)


@lru_cache(maxsize=10000)
def _cached_math_equal(pred_str: str, gt_str: str) -> bool:
    """Cached version of math_equal."""
    try:
        return math_equal(pred_str, gt_str, include_percentage=True, is_close=True)
    except Exception:
        return False


@lru_cache(maxsize=10000)
def _cached_grade_answer(pred_str: str, gt_str: str) -> bool:
    """Cached version of grade_answer."""
    try:
        return grade_answer(pred_str, gt_str)
    except Exception:
        return False


def is_correct_with_math_verify(predicted: Any, groundtruth: Any) -> bool:
    """
    Check correctness using MATH folder evaluation approach.
    Tries both grader.math_equal and verify_utils.grade_answer for robustness.
    """
    if predicted is None or groundtruth is None:
        return False

    pred_str = str(predicted).strip()
    gt_str = str(groundtruth).strip()

    if not pred_str or not gt_str:
        return False

    # Try cached math_equal first (most comprehensive)
    if _cached_math_equal(pred_str, gt_str):
        return True

    # Fallback to cached grade_answer
    if _cached_grade_answer(pred_str, gt_str):
        return True

    return False


def majority_vote(
    extracted_answers: List[Optional[str]],
    groundtruth: str,
) -> bool:
    """
    Perform majority vote on extracted answers from multiple trees (m trees).
    Returns True if the most common answer is correct.
    """
    valid_answers = [a for a in extracted_answers if a is not None and str(a).strip() != ""]
    if not valid_answers:
        return False

    norm_to_original: dict[str, str] = {}
    norm_list: List[str] = []

    for ans in valid_answers:
        # Try to normalize answer for better matching
        try:
            # Use extract_answer to normalize
            normalized = extract_answer(str(ans), "math")
            key = normalized if normalized else str(ans).strip()
        except Exception:
            key = str(ans).strip()

        norm_list.append(key)
        if key not in norm_to_original:
            norm_to_original[key] = ans   # store first seen original

    if not norm_list:
        return False

    # Get most common answer
    counter = Counter(norm_list)
    best_key, _ = counter.most_common(1)[0]
    best_ans = norm_to_original[best_key]

    return is_correct_with_math_verify(best_ans, groundtruth)


data_name = "math"  # Dataset name for answer extraction

seeds    = range(5)
base_root = (
    "/prj/corp/airesearch/lasvegas/vol11-scratch/nluu/baseline-tjts/src/output/AIME24_beam_search/Qwen2.5-3B-Instruct/Qwen2.5-Math-PRM-7B"
)


def process_question_dvts(question_dir_path: Tuple[str, str]) -> Tuple[int, bool]:
    """
    Process a single question directory.
    Load m trees (m jsonl files), perform majority vote, and return result.
    Returns (1 if groundtruth found, boolean result).
    """
    question_dir, base_dir = question_dir_path
    question_path = os.path.join(base_dir, question_dir)

    try:
        extracted_answers: List[Optional[str]] = []
        problem_str: Optional[str] = None
        groundtruth: Optional[str] = None

        # ── collect answers from all m trees (m jsonl files) ──────────────────────────
        jsonl_files = [f for f in os.listdir(question_path) if f.endswith(".jsonl")]

        for filename in jsonl_files:
            file_path = os.path.join(question_path, filename)
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line)

                        problem_str = data.get("question", None)
                        groundtruth = data.get("groundtruth", None)

                        for out in data.get("output", []):
                            ans = out.get("extracted_answer", None)
                            extracted_answers.append(ans)
            except Exception:
                continue

        # ── majority vote + evaluation ────────────────────────────
        if groundtruth is None:
            return 0, False

        # Extract and normalize ground truth answer
        gt_answer = extract_answer(groundtruth, data_name)
        if not gt_answer:
            return 0, False

        # Perform majority vote on extracted answers from m trees
        result = majority_vote(extracted_answers, gt_answer)
        return 1, result

    except Exception:
        return 0, False


for seed in seeds:
    base_dir = f"{base_root}/seed_{seed}_width_16_num_seq_2_num_q_4"
    print(f"Processing: {base_dir}")

    # Get all question directories
    question_dirs = [d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d))]

    if not question_dirs:
        print("No question directories found!")
        continue

    acc_sum = 0
    total = 0

    # Use multiprocessing to process questions in parallel
    num_workers = max(1, cpu_count() - 1)

    with Pool(processes=num_workers) as pool:
        # Create list of (question_dir, base_dir) tuples for workers
        tasks = [(q_dir, base_dir) for q_dir in question_dirs]

        # Process with progress bar
        for valid_count, result in tqdm(
            pool.imap_unordered(process_question_dvts, tasks),
            total=len(tasks),
            desc="Processing questions",
            unit="q",
            smoothing=0.1
        ):
            total += valid_count
            acc_sum += int(result)

    acc = acc_sum / total if total > 0 else 0.0

    print(f"\nSeed {seed}")
    print(f"Total questions : {total}")
    print(f"ACC             : {acc:.4f}")
    print("-" * 40)