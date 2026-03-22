"""
Comprehensive voting evaluation for MINERVA questions.
Applies all voting strategies from vote_utils.py and generates avg_result.json
"""

import os
import json
import re
from collections import Counter, defaultdict
from typing import List, Optional, Dict, Tuple
from pathlib import Path

# ── Voting Functions (from vote_utils.py) ──────────────────────────────────

MAJORITY_VOTE = "majority_vote"
PRM_MIN_MAX = "prm_min_max"
PRM_MIN_VOTE = "prm_min_vote"
PRM_LAST_MAX = "prm_last_max"
PRM_LAST_VOTE = "prm_last_vote"
PRM_AVG_MAX = "prm_avg_max"
PRM_AVG_VOTE = "prm_avg_vote"

VOTING_STRATEGIES = [
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_MAX,
    PRM_LAST_VOTE,
    PRM_AVG_MAX,
    PRM_AVG_VOTE,
]


def _agg_majority_vote(x_list: List[str], unused_v_list, return_reward=False):
    """Vote based on answer frequency."""
    counts = Counter(x_list)
    most_common = max(counts, key=counts.get)
    if return_reward:
        return most_common, [0.0]
    return most_common


def _agg_orm_vote(x_list: List[str], v_list: List[float], return_reward_idx=False):
    """Weighted vote by summing scores for each answer."""
    assert len(x_list) == len(v_list)
    x_dict = defaultdict(lambda: 0.0)
    for x, v in zip(x_list, v_list):
        x_dict[x] += v

    highest_x = max(x_dict, key=x_dict.get)
    if return_reward_idx:
        idx_list = [i for i, x in enumerate(x_list) if x == highest_x]
        corresponding_v_list = [v_list[idx] for idx in idx_list]
        idx = corresponding_v_list.index(max(corresponding_v_list))
        return highest_x, idx
    return highest_x


def _agg_prm_min_max(x_list: List[str], v_list: List[List[float]], return_reward=False):
    """Select output with highest minimum reward score."""
    new_v_list = [min(v) if v else -1.0 for v in v_list]
    idx = new_v_list.index(max(new_v_list))
    text_max = x_list[idx]
    if return_reward:
        return text_max, v_list[idx]
    return text_max


def _agg_prm_last_max(x_list: List[str], v_list: List[List[float]], return_reward=False):
    """Select output with highest last reward score."""
    new_v_list = [v[-1] if v else -1.0 for v in v_list]
    idx = new_v_list.index(max(new_v_list))
    text_max = x_list[idx]
    if return_reward:
        return text_max, v_list[idx]
    return text_max


def _agg_prm_min_vote(x_list: List[str], v_list: List[List[float]], return_reward=False):
    """Vote with weights = minimum reward of each output."""
    new_v_list = [min(v) if v else -1.0 for v in v_list]
    if return_reward:
        x, idx = _agg_orm_vote(x_list, new_v_list, return_reward_idx=True)
        return x, v_list[idx]
    return _agg_orm_vote(x_list, new_v_list)


def _agg_prm_last_vote(x_list: List[str], v_list: List[List[float]], return_reward=False):
    """Vote with weights = last reward of each output."""
    new_v_list = [v[-1] if v else -1.0 for v in v_list]
    if return_reward:
        x, idx = _agg_orm_vote(x_list, new_v_list, return_reward_idx=True)
        return x, v_list[idx]
    return _agg_orm_vote(x_list, new_v_list)


def _agg_prm_avg_max(x_list: List[str], v_list: List[List[float]], return_reward=False):
    """Select output with highest average reward score."""
    new_v_list = [(sum(v) / len(v)) if v else -1.0 for v in v_list]
    idx = new_v_list.index(max(new_v_list))
    text_max = x_list[idx]
    if return_reward:
        return text_max, v_list[idx]
    return text_max


def _agg_prm_avg_vote(x_list: List[str], v_list: List[List[float]], return_reward=False):
    """Vote with weights = average reward of each output."""
    new_v_list = [(sum(v) / len(v)) if v else -1.0 for v in v_list]
    if return_reward:
        x, idx = _agg_orm_vote(x_list, new_v_list, return_reward_idx=True)
        return x, v_list[idx]
    return _agg_orm_vote(x_list, new_v_list)


AGG_FN_MAP = {
    MAJORITY_VOTE: _agg_majority_vote,
    PRM_MIN_MAX: _agg_prm_min_max,
    PRM_MIN_VOTE: _agg_prm_min_vote,
    PRM_LAST_MAX: _agg_prm_last_max,
    PRM_LAST_VOTE: _agg_prm_last_vote,
    PRM_AVG_MAX: _agg_prm_avg_max,
    PRM_AVG_VOTE: _agg_prm_avg_vote,
}

# ── Answer verification (from evaluate_minerva_questions.py) ─────────────

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


def clean_latex(text: str) -> str:
    """Remove LaTeX formatting from text."""
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\left\(|\\\right\)', '', text)
    text = re.sub(r'\\left\{|\\\right\}', '', text)
    text = re.sub(r'\\times', '*', text)
    text = re.sub(r'\^', '^', text)
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_numeric_value(text: str) -> Optional[float]:
    """Try to extract and evaluate numeric value from text."""
    text = clean_latex(text)

    # Remove common units
    units_pattern = r'\s*(?:cm|m|mm|kg|g|s|Hz|arcsec|arcmin|degree|ergs?/s|erg/s|Angstroms?|K|C|rad|°|″|′)(?:\s|$|/)'
    text_no_units = re.sub(units_pattern, '', text)

    try:
        return float(text_no_units)
    except ValueError:
        pass

    # Parse fractions
    try:
        match = re.search(r'\(?(\d+)\)?/\(?(\d+)\)?', text_no_units)
        if match:
            numerator = int(match.group(1))
            denominator = int(match.group(2))
            return numerator / denominator
    except (ValueError, AttributeError, ZeroDivisionError):
        pass

    # Parse scientific notation
    try:
        match = re.search(r'([-+]?\d*\.?\d+)\s*(?:\*|x)\s*10\^?\{?([+-]?\d+)\}?', text_no_units, re.IGNORECASE)
        if match:
            base = float(match.group(1))
            exp = int(match.group(2))
            return base * (10 ** exp)
    except (ValueError, AttributeError):
        pass

    # Handle "1.4e31" format
    try:
        match = re.search(r'([-+]?\d*\.?\d+)e([+-]?\d+)', text_no_units, re.IGNORECASE)
        if match:
            base = float(match.group(1))
            exp = int(match.group(2))
            return base * (10 ** exp)
    except (ValueError, AttributeError):
        pass

    return None


def verify_answer(response: str, ground_truth: str, tolerance: float = 0.02) -> bool:
    """Verify response against ground truth with smart comparison."""
    pred = extract_boxed_answer(response)
    if pred is None:
        return False

    pred = pred.strip()
    gt = ground_truth.strip()

    # Direct string match
    if pred == gt:
        return True

    # Try numeric comparison
    pred_val = extract_numeric_value(pred)
    gt_val = extract_numeric_value(gt)

    if pred_val is not None and gt_val is not None:
        abs_diff = abs(pred_val - gt_val)
        abs_gt = abs(gt_val)

        if abs_gt < 1e-10:
            return abs_diff < 1e-5
        elif abs_gt < 0.1:
            relative_error = abs_diff / abs_gt if abs_gt != 0 else float('inf')
            return relative_error < tolerance
        else:
            relative_error = abs_diff / abs_gt if abs_gt != 0 else float('inf')
            return relative_error < tolerance

    # Clean LaTeX and try string match
    pred_clean = clean_latex(pred)
    gt_clean = clean_latex(gt)

    if pred_clean == gt_clean:
        return True

    # Try matching after removing spaces/special chars
    pred_alphanum = re.sub(r'[^\w]', '', pred_clean).lower()
    gt_alphanum = re.sub(r'[^\w]', '', gt_clean).lower()

    if pred_alphanum == gt_alphanum and len(pred_alphanum) > 0:
        return True

    return False


# ── Main evaluation ────────────────────────────────────────────────────────

def evaluate_all_strategies():
    base_dir = Path("/Users/luungoc/Project/compute-optimal-tts/MINERVA_best_of_n")
    base_dir = base_dir / "Qwen2.5-7B-Instruct" / "Qwen2.5-Math-PRM-7B" / "seed_0_width_32_num_seq_32_num_q_0"

    # Get all question directories
    question_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("question_")],
                          key=lambda x: int(x.name.split("_")[1]))

    print(f"Found {len(question_dirs)} question directories")
    print("=" * 80)

    # Initialize tracking
    strategy_results = {strategy: {'correct': 0, 'total': 0} for strategy in VOTING_STRATEGIES}
    total_completion_tokens = 0

    # Process each question
    for idx, question_dir in enumerate(question_dirs):
        record_path = question_dir / "record_0.jsonl"

        if not record_path.exists():
            continue

        try:
            with open(record_path, 'r', encoding='utf-8') as f:
                line = f.readline().strip()
                if not line:
                    continue
                record = json.loads(line)
        except Exception as e:
            print(f"[{idx}] {question_dir.name}: Error reading JSON - {e}")
            continue

        # Extract data
        question = record.get('question', '')
        ground_truth = record.get('groundtruth', '')
        outputs = record.get('output', [])

        if not outputs or not ground_truth:
            continue

        # Prepare data for voting
        x_list = [o.get('extracted_answer') for o in outputs]
        v_list = [o.get('reward_history', []) for o in outputs]

        # Count tokens
        total_completion_tokens += record.get('result', {}).get('total_completion_tokens', 0)

        # Apply each voting strategy
        for strategy in VOTING_STRATEGIES:
            try:
                if strategy == MAJORITY_VOTE:
                    # Majority vote doesn't use reward scores
                    selected_answer = AGG_FN_MAP[strategy](x_list, v_list)
                else:
                    # PRM-based strategies use reward histories
                    selected_answer = AGG_FN_MAP[strategy](x_list, v_list)

                if selected_answer:
                    # Verify answer
                    is_correct = verify_answer(f"\\boxed{{{selected_answer}}}", ground_truth)
                else:
                    is_correct = False

                strategy_results[strategy]['total'] += 1
                if is_correct:
                    strategy_results[strategy]['correct'] += 1

            except Exception as e:
                # Skip if voting fails
                pass

        # Print progress
        if (idx + 1) % 50 == 0 or (idx + 1) == len(question_dirs):
            print(f"[{idx+1}/{len(question_dirs)}]")

    # Calculate accuracies
    results_dict = {}
    for strategy in VOTING_STRATEGIES:
        total = strategy_results[strategy]['total']
        correct = strategy_results[strategy]['correct']
        accuracy = correct / total if total > 0 else 0.0
        results_dict[strategy] = accuracy
        print(f"{strategy:20} {correct:3d}/{total} = {accuracy*100:6.2f}%")

    # Calculate average tokens
    total_questions = strategy_results[MAJORITY_VOTE]['total']
    avg_tokens = total_completion_tokens / total_questions if total_questions > 0 else 0

    results_dict['total_completion_tokens'] = avg_tokens

    print("\n" + "=" * 80)
    print(f"Average tokens per question: {avg_tokens:.2f}")
    print("=" * 80)

    # Save results to JSON
    output_file = base_dir / "avg_result.json"
    with open(output_file, 'w') as f:
        json.dump([results_dict], f)

    print(f"\nResults saved to: {output_file}")
    return results_dict


if __name__ == "__main__":
    evaluate_all_strategies()
