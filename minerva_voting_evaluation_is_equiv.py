"""
Comprehensive voting evaluation for MINERVA questions using is_equiv (SymPy).
Applies all voting strategies from vote_utils.py with mathematical equivalence.
Uses is_equiv() from eval_github.py for mathematical equivalence checking.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from typing import List, Optional
from pathlib import Path
import signal

import sympy
from sympy.parsing.latex import parse_latex

# Setup logging
eval_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

# ── Timeout Context Manager ─────────────────────────────────────────────────

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


# ── is_equiv Implementation (from eval_github.py) ───────────────────────────

def is_equiv(x1: str, x2: str) -> bool:
    """
    SymPy-based mathematical equivalence check.
    x1 and x2 are normalized latex strings.
    Parses LaTeX, simplifies mathematically, checks if difference = 0.
    """
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
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                eval_logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False

    return False


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize answer following Lewkowycz et al. (2022) from eval_github.py
    """
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
        "square",
        "ways",
        "integers",
        "dollars",
        "mph",
        "inches",
        "ft",
        "hours",
        "km",
        "units",
        "\\ldots",
        "sue",
        "points",
        "feet",
        "minutes",
        "digits",
        "cents",
        "degrees",
        "cm",
        "gm",
        "pounds",
        "meters",
        "meals",
        "edges",
        "students",
        "childrentickets",
        "multiples",
        "\\text{s}",
        "\\text{.}",
        "\\text{\ns}",
        "\\text{}^2",
        "\\text{}^3",
        "\\text{\n}",
        "\\text{}",
        r"\mathrm{th}",
        r"^\circ",
        r"^{\circ}",
        r"\;",
        r",\!",
        "{,}",
        '"',
        "\\dots",
    ]

    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


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


def evaluate_all_strategies_is_equiv():
    """Evaluate using is_equiv (SymPy mathematical equivalence)."""
    base_dir = Path("/Users/luungoc/Project/compute-optimal-tts/MINERVA_best_of_n")
    base_dir = base_dir / "Qwen2.5-7B-Instruct" / "Qwen2.5-Math-PRM-7B" / "seed_0_width_32_num_seq_32_num_q_0"

    # Get all question directories
    question_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("question_")],
        key=lambda x: int(x.name.split("_")[1])
    )

    print(f"Found {len(question_dirs)} question directories")
    print("=" * 80)
    print("USING: is_equiv (SymPy mathematical equivalence from eval_github.py)")
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
        except Exception:
            continue

        # Extract data
        ground_truth = record.get('groundtruth', '')
        outputs = record.get('output', [])

        if not outputs or not ground_truth:
            continue

        # Normalize ground truth
        normalized_gt = normalize_final_answer(ground_truth)

        # Prepare data for voting
        x_list = [o.get('extracted_answer') for o in outputs]
        v_list = [o.get('reward_history', []) for o in outputs]

        # Count tokens
        total_completion_tokens += record.get('result', {}).get('total_completion_tokens', 0)

        # Apply each voting strategy
        for strategy in VOTING_STRATEGIES:
            try:
                if strategy == MAJORITY_VOTE:
                    selected_answer = AGG_FN_MAP[strategy](x_list, v_list)
                else:
                    selected_answer = AGG_FN_MAP[strategy](x_list, v_list)

                if selected_answer:
                    # Normalize the selected answer
                    normalized_answer = normalize_final_answer(selected_answer)

                    # Verify using is_equiv (mathematical equivalence)
                    is_correct = is_equiv(normalized_answer, normalized_gt)
                else:
                    is_correct = False

                strategy_results[strategy]['total'] += 1
                if is_correct:
                    strategy_results[strategy]['correct'] += 1

            except Exception:
                pass

        # Print progress
        if (idx + 1) % 50 == 0 or (idx + 1) == len(question_dirs):
            print(f"[{idx+1}/{len(question_dirs)}]")

    # Calculate accuracies
    results_dict = {}
    print()
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
    output_file = base_dir / "avg_result_is_equiv.json"
    with open(output_file, 'w') as f:
        json.dump([results_dict], f)

    print(f"\nResults saved to: {output_file}")
    return results_dict


if __name__ == "__main__":
    evaluate_all_strategies_is_equiv()
