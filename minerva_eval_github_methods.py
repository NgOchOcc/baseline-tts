"""
Evaluation using eval_github.py methods.
Implements two verification approaches:
1. is_equiv() - SymPy based mathematical equivalence (sympy.simplify)
2. math_verify() - Advanced verification library

For each output index (0-31), calculate accuracy using both methods.
"""

import json
import logging
from pathlib import Path
from typing import Optional
import re
import signal

import sympy
from sympy.parsing.latex import parse_latex

# Setup logging
eval_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

# Try importing math_verify, but make it optional
try:
    from math_verify import parse, verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False
    eval_logger.warning("math_verify not available, using only is_equiv method")

# Import answer extraction from minerva_utils
from minerva_utils import extract_boxed_answer


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


def verify_with_math_verify(candidate: str, ground_truth: str) -> bool:
    """
    Use math_verify library for verification.
    Parses both candidate and ground_truth, then uses verify().
    """
    if not HAS_MATH_VERIFY:
        return False

    try:
        # Parse ground truth with LaTeX extraction
        gold_parsed = parse(
            f"\\boxed{{{ground_truth}}}",
            extraction_config=[]
        )

        # Parse candidate response
        cand_parsed = parse(candidate, extraction_config=[])

        # Verify equivalence
        result = verify(gold=gold_parsed, target=cand_parsed)
        return bool(result)
    except Exception as e:
        eval_logger.debug(f"math_verify failed: {e}")
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


def evaluate_all_indices_github():
    """Evaluate using eval_github.py methods for all output indices (0-31)."""
    base_dir = Path("/Users/luungoc/Project/compute-optimal-tts/MINERVA_best_of_n")
    base_dir = base_dir / "Qwen2.5-7B-Instruct" / "Qwen2.5-Math-PRM-7B" / "seed_0_width_32_num_seq_32_num_q_0"

    # Get all question directories
    question_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("question_")],
        key=lambda x: int(x.name.split("_")[1])
    )

    print(f"Found {len(question_dirs)} question directories")
    print("=" * 120)
    print("BASELINE EVALUATION: Using eval_github.py methods (is_equiv + math_verify)")
    print("=" * 120)
    print()

    # Track results for each output index and method
    index_results = {}
    for idx in range(32):
        index_results[idx] = {
            'is_equiv': {'correct': 0, 'total': 0},
            'math_verify': {'correct': 0, 'total': 0},
        }

    detailed_results_per_index = {idx: {'is_equiv': [], 'math_verify': []} for idx in range(32)}

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

        # Test each output index
        for output_idx in range(min(32, len(outputs))):
            try:
                output_item = outputs[output_idx]
                baseline_answer = output_item.get('extracted_answer')

                if not baseline_answer:
                    continue

                # Normalize the ground truth answer
                normalized_gt = normalize_final_answer(ground_truth)
                normalized_answer = normalize_final_answer(baseline_answer)

                # Method 1: is_equiv (SymPy)
                try:
                    is_correct_equiv = is_equiv(normalized_answer, normalized_gt)
                except Exception:
                    is_correct_equiv = False

                index_results[output_idx]['is_equiv']['total'] += 1
                if is_correct_equiv:
                    index_results[output_idx]['is_equiv']['correct'] += 1

                detailed_results_per_index[output_idx]['is_equiv'].append({
                    'question_idx': q_idx,
                    'answer': baseline_answer,
                    'is_correct': is_correct_equiv,
                })

                # Method 2: math_verify
                try:
                    is_correct_mv = verify_with_math_verify(
                        f"\\boxed{{{baseline_answer}}}",
                        ground_truth
                    )
                except Exception:
                    is_correct_mv = False

                index_results[output_idx]['math_verify']['total'] += 1
                if is_correct_mv:
                    index_results[output_idx]['math_verify']['correct'] += 1

                detailed_results_per_index[output_idx]['math_verify'].append({
                    'question_idx': q_idx,
                    'answer': baseline_answer,
                    'is_correct': is_correct_mv,
                })

            except (IndexError, KeyError, TypeError):
                pass

        # Print progress
        if (q_idx + 1) % 50 == 0 or (q_idx + 1) == len(question_dirs):
            print(f"[{q_idx+1}/{len(question_dirs)}] Processed")

    # Print results for is_equiv
    print("\n" + "=" * 120)
    print("RESULTS: is_equiv (SymPy-based mathematical equivalence)")
    print("=" * 120)
    print()
    print(f"{'Index':<8} {'Correct':<10} {'Total':<10} {'Accuracy':<12}")
    print("-" * 120)

    results_equiv = []
    for idx in range(32):
        total = index_results[idx]['is_equiv']['total']
        correct = index_results[idx]['is_equiv']['correct']
        accuracy = (correct / total * 100) if total > 0 else 0.0
        results_equiv.append((idx, correct, total, accuracy))

    results_equiv.sort(key=lambda x: x[3], reverse=True)
    for idx, correct, total, accuracy in results_equiv[:15]:
        if total > 0:
            print(f"{idx:<8} {correct:<10} {total:<10} {accuracy:>6.2f}%")

    print()
    best_equiv = results_equiv[0]
    print(f"BEST (is_equiv): Index {best_equiv[0]} with {best_equiv[3]:.2f}% accuracy ({best_equiv[1]}/{best_equiv[2]})")

    # Log detailed results for best index
    print()
    print("=" * 120)
    print(f"DETAILED RESULTS: is_equiv at Index {best_equiv[0]}")
    print("=" * 120)
    print()
    print(f"{'Q_ID':<6} {'Ground Truth':<30} {'Final Answer':<30} {'Match':<8}")
    print("-" * 120)

    best_idx = best_equiv[0]
    if best_idx in detailed_results_per_index:
        for i, result in enumerate(detailed_results_per_index[best_idx]['is_equiv'][:20]):
            q_id = result['question_idx']
            answer = result['answer'][:28] if result['answer'] else "N/A"
            match = "✓ PASS" if result['is_correct'] else "✗ FAIL"
            # Find ground truth for this question
            gt_answer = "N/A"
            for q_idx, q_dir in enumerate(question_dirs):
                if q_idx == q_id:
                    record_path = q_dir / "record_0.jsonl"
                    try:
                        with open(record_path, 'r', encoding='utf-8') as f:
                            line = f.readline().strip()
                            if line:
                                record = json.loads(line)
                                gt_answer = record.get('groundtruth', 'N/A')[:28]
                    except:
                        pass
                    break
            print(f"{q_id:<6} {gt_answer:<30} {answer:<30} {match:<8}")
        print()

    # Print results for math_verify
    print("\n" + "=" * 120)
    print("RESULTS: math_verify (Advanced verification library)")
    print("=" * 120)
    print()
    print(f"{'Index':<8} {'Correct':<10} {'Total':<10} {'Accuracy':<12}")
    print("-" * 120)

    results_mv = []
    for idx in range(32):
        total = index_results[idx]['math_verify']['total']
        correct = index_results[idx]['math_verify']['correct']
        accuracy = (correct / total * 100) if total > 0 else 0.0
        results_mv.append((idx, correct, total, accuracy))

    results_mv.sort(key=lambda x: x[3], reverse=True)
    for idx, correct, total, accuracy in results_mv[:15]:
        if total > 0:
            print(f"{idx:<8} {correct:<10} {total:<10} {accuracy:>6.2f}%")

    print()
    best_mv = results_mv[0]
    print(f"BEST (math_verify): Index {best_mv[0]} with {best_mv[3]:.2f}% accuracy ({best_mv[1]}/{best_mv[2]})")

    # Log detailed results for best index
    print()
    print("=" * 120)
    print(f"DETAILED RESULTS: math_verify at Index {best_mv[0]}")
    print("=" * 120)
    print()
    print(f"{'Q_ID':<6} {'Ground Truth':<30} {'Final Answer':<30} {'Match':<8}")
    print("-" * 120)

    best_idx_mv = best_mv[0]
    if best_idx_mv in detailed_results_per_index:
        for i, result in enumerate(detailed_results_per_index[best_idx_mv]['math_verify'][:20]):
            q_id = result['question_idx']
            answer = result['answer'][:28] if result['answer'] else "N/A"
            match = "✓ PASS" if result['is_correct'] else "✗ FAIL"
            # Find ground truth for this question
            gt_answer = "N/A"
            for q_idx, q_dir in enumerate(question_dirs):
                if q_idx == q_id:
                    record_path = q_dir / "record_0.jsonl"
                    try:
                        with open(record_path, 'r', encoding='utf-8') as f:
                            line = f.readline().strip()
                            if line:
                                record = json.loads(line)
                                gt_answer = record.get('groundtruth', 'N/A')[:28]
                    except:
                        pass
                    break
            print(f"{q_id:<6} {gt_answer:<30} {answer:<30} {match:<8}")
        print()

    print("\n" + "=" * 120)

    # Save detailed results
    output_file = base_dir / "baseline_result_all_indices_github_methods.json"
    with open(output_file, 'w') as f:
        json.dump({
            'is_equiv_summary': {
                f'index_{idx}': {
                    'correct': index_results[idx]['is_equiv']['correct'],
                    'total': index_results[idx]['is_equiv']['total'],
                    'accuracy': (index_results[idx]['is_equiv']['correct'] / index_results[idx]['is_equiv']['total']
                                if index_results[idx]['is_equiv']['total'] > 0 else 0.0)
                }
                for idx in range(32)
            },
            'math_verify_summary': {
                f'index_{idx}': {
                    'correct': index_results[idx]['math_verify']['correct'],
                    'total': index_results[idx]['math_verify']['total'],
                    'accuracy': (index_results[idx]['math_verify']['correct'] / index_results[idx]['math_verify']['total']
                                if index_results[idx]['math_verify']['total'] > 0 else 0.0)
                }
                for idx in range(32)
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")
    print()

    return index_results


if __name__ == "__main__":
    evaluate_all_indices_github()
