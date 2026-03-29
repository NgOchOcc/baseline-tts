"""
Comprehensive MINERVA evaluation using is_equiv (SymPy mathematical equivalence).
Saves three output files:
1. voting_results.json - Voting strategy results (like avg_result.json)
2. pass_at_k_results.json - pass@1, pass@8, pass@16, pass@32 metrics
3. detailed_samples.json - Full details for each sample (ground_truth, predict, result)
"""

import json
import logging
import re
import signal
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import sympy
from sympy.parsing.latex import parse_latex

# Setup logging
logging.basicConfig(level=logging.WARNING)
eval_logger = logging.getLogger(__name__)


class timeout:
    """Timeout context manager for preventing infinite parsing loops."""
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """SymPy-based mathematical equivalence check."""
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                return False

            try:
                return sympy.simplify(diff) == 0
            except ValueError:
                return False
    except TimeoutError:
        return False
    except Exception:
        return False


def normalize_final_answer(final_answer: str) -> str:
    """Normalize answer following Lewkowycz et al. (2022)."""
    if not final_answer:
        return ""

    SUBSTITUTIONS = [
        ("an ", ""),
        ("a ", ""),
        (".$", "$"),
        ("\\$", ""),
        (r"\ ", ""),
        (" ", ""),
        ("mbox", "text"),
        (",\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
    ]
    REMOVED_EXPRESSIONS = [
        "square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours",
        "km", "units", "\\ldots", "sue", "points", "feet", "minutes", "digits",
        "cents", "degrees", "cm", "gm", "pounds", "meters", "meals", "edges",
        "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}",
        "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
        r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
    ]

    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


# ── Voting Functions ────────────────────────────────────────────────────────

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
    counts = Counter(x_list)
    most_common = max(counts, key=counts.get)
    if return_reward:
        return most_common, [0.0]
    return most_common


def _agg_orm_vote(x_list: List[str], v_list: List[float], return_reward_idx=False):
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
    new_v_list = [min(v) if v else -1.0 for v in v_list]
    idx = new_v_list.index(max(new_v_list))
    text_max = x_list[idx]
    if return_reward:
        return text_max, v_list[idx]
    return text_max


def _agg_prm_last_max(x_list: List[str], v_list: List[List[float]], return_reward=False):
    new_v_list = [v[-1] if v else -1.0 for v in v_list]
    idx = new_v_list.index(max(new_v_list))
    text_max = x_list[idx]
    if return_reward:
        return text_max, v_list[idx]
    return text_max


def _agg_prm_min_vote(x_list: List[str], v_list: List[List[float]], return_reward=False):
    new_v_list = [min(v) if v else -1.0 for v in v_list]
    if return_reward:
        x, idx = _agg_orm_vote(x_list, new_v_list, return_reward_idx=True)
        return x, v_list[idx]
    return _agg_orm_vote(x_list, new_v_list)


def _agg_prm_last_vote(x_list: List[str], v_list: List[List[float]], return_reward=False):
    new_v_list = [v[-1] if v else -1.0 for v in v_list]
    if return_reward:
        x, idx = _agg_orm_vote(x_list, new_v_list, return_reward_idx=True)
        return x, v_list[idx]
    return _agg_orm_vote(x_list, new_v_list)


def _agg_prm_avg_max(x_list: List[str], v_list: List[List[float]], return_reward=False):
    new_v_list = [(sum(v) / len(v)) if v else -1.0 for v in v_list]
    idx = new_v_list.index(max(new_v_list))
    text_max = x_list[idx]
    if return_reward:
        return text_max, v_list[idx]
    return text_max


def _agg_prm_avg_vote(x_list: List[str], v_list: List[List[float]], return_reward=False):
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


def evaluate_is_equiv():
    """Main evaluation using is_equiv with voting strategies and pass@k metrics."""
    base_dir = Path("/Users/luungoc/Project/compute-optimal-tts/MINERVA_best_of_n")
    base_dir = base_dir / "Qwen2.5-7B-Instruct" / "Qwen2.5-Math-PRM-7B" / "seed_0_width_32_num_seq_32_num_q_0"

    # Get all question directories
    question_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("question_")],
        key=lambda x: int(x.name.split("_")[1])
    )

    print(f"Found {len(question_dirs)} question directories")
    print("=" * 80)
    print("EVALUATING with is_equiv (SymPy mathematical equivalence)")
    print("=" * 80)
    print()

    # Initialize tracking
    strategy_results = {strategy: {'correct': 0, 'total': 0} for strategy in VOTING_STRATEGIES}
    pass_at_k = {1: {'correct': 0, 'total': 0}, 8: {'correct': 0, 'total': 0}, 16: {'correct': 0, 'total': 0}, 32: {'correct': 0, 'total': 0}}
    all_samples = []
    total_completion_tokens = 0

    # Process each question
    for q_idx, question_dir in enumerate(question_dirs):
        record_path = question_dir / "record_0.jsonl"

        if not record_path.exists():
            continue

        try:
            with open(record_path, 'r', encoding='utf-8') as f:
                line = f.readline().strip()
                if not line:
                    continue
                record = json.loads(line)
        except Exception:
            continue

        # Extract data
        ground_truth = record.get('groundtruth', '')
        outputs = record.get('output', [])

        if not outputs or not ground_truth:
            continue

        normalized_gt = normalize_final_answer(ground_truth)
        x_list = [o.get('extracted_answer') for o in outputs]
        v_list = [o.get('reward_history', []) for o in outputs]

        total_completion_tokens += record.get('result', {}).get('total_completion_tokens', 0)

        # === VOTING STRATEGIES ===
        for strategy in VOTING_STRATEGIES:
            try:
                selected_answer = AGG_FN_MAP[strategy](x_list, v_list)

                if selected_answer:
                    normalized_answer = normalize_final_answer(selected_answer)
                    is_correct = is_equiv(normalized_answer, normalized_gt)
                else:
                    is_correct = False

                strategy_results[strategy]['total'] += 1
                if is_correct:
                    strategy_results[strategy]['correct'] += 1

            except Exception:
                pass

        # === PASS@K METRICS ===
        for k in [1, 8, 16, 32]:
            # Check if any of first k outputs has correct answer
            is_correct_at_k = False
            for i in range(min(k, len(x_list))):
                answer = x_list[i]
                if answer:
                    normalized_answer = normalize_final_answer(answer)
                    if is_equiv(normalized_answer, normalized_gt):
                        is_correct_at_k = True
                        break

            pass_at_k[k]['total'] += 1
            if is_correct_at_k:
                pass_at_k[k]['correct'] += 1

        # === DETAILED SAMPLES ===
        for output_idx in range(min(32, len(x_list))):
            answer = x_list[output_idx]
            if answer:
                normalized_answer = normalize_final_answer(answer)
                is_correct = is_equiv(normalized_answer, normalized_gt)
                all_samples.append({
                    'question_idx': q_idx,
                    'output_index': output_idx,
                    'ground_truth': ground_truth,
                    'normalized_ground_truth': normalized_gt,
                    'predicted_answer': answer,
                    'normalized_predicted_answer': normalized_answer,
                    'is_correct': is_correct,
                })

        # Print progress
        if (q_idx + 1) % 50 == 0 or (q_idx + 1) == len(question_dirs):
            print(f"[{q_idx+1}/{len(question_dirs)}] Processed")

    # === SAVE RESULTS ===

    # 1. Voting results (like avg_result.json)
    voting_results = {}
    print()
    print("=" * 80)
    print("VOTING STRATEGY RESULTS")
    print("=" * 80)
    for strategy in VOTING_STRATEGIES:
        total = strategy_results[strategy]['total']
        correct = strategy_results[strategy]['correct']
        accuracy = correct / total if total > 0 else 0.0
        voting_results[strategy] = accuracy
        print(f"{strategy:20} {correct:3d}/{total} = {accuracy*100:6.2f}%")

    avg_tokens = total_completion_tokens / len(question_dirs) if len(question_dirs) > 0 else 0
    voting_results['total_completion_tokens'] = avg_tokens

    voting_file = base_dir / "voting_results.json"
    with open(voting_file, 'w') as f:
        json.dump([voting_results], f, indent=2)
    print(f"\n✓ Saved to: {voting_file}")

    # 2. Pass@K results
    pass_at_k_results = {}
    print()
    print("=" * 80)
    print("PASS@K METRICS")
    print("=" * 80)
    for k in [1, 8, 16, 32]:
        total = pass_at_k[k]['total']
        correct = pass_at_k[k]['correct']
        accuracy = correct / total if total > 0 else 0.0
        pass_at_k_results[f'pass@{k}'] = accuracy
        print(f"pass@{k:<2} {correct:3d}/{total} = {accuracy*100:6.2f}%")

    pass_at_k_file = base_dir / "pass_at_k_results.json"
    with open(pass_at_k_file, 'w') as f:
        json.dump(pass_at_k_results, f, indent=2)
    print(f"\n✓ Saved to: {pass_at_k_file}")

    # 3. Detailed samples
    detailed_file = base_dir / "detailed_samples.json"
    with open(detailed_file, 'w') as f:
        json.dump({
            'total_samples': len(all_samples),
            'samples': all_samples
        }, f, indent=2)
    print()
    print("=" * 80)
    print(f"✓ Saved detailed samples ({len(all_samples)} samples) to: {detailed_file}")
    print("=" * 80)

    return voting_results, pass_at_k_results, all_samples


if __name__ == "__main__":
    evaluate_is_equiv()
