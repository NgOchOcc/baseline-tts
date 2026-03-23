"""
MINERVA task evaluation module using math_verify or fallback to MATH folder functions.
Adapted from original.py (LM Evaluation Harness) and test_minerva.py
"""

import logging
import re
import signal
from typing import Optional
from importlib.metadata import version

from envs.base_env import CoTEnv, NoLegalActionException, INVALID_ANS

logger = logging.getLogger(__name__)

# Try to import math_verify and sympy for comprehensive evaluation
try:
    from math_verify import verify, parse
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    HAS_MATH_VERIFY = True
except ImportError:
    print("⚠️  math_verify not available, using fallback methods")
    HAS_MATH_VERIFY = False

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    HAS_SYMPY = True
except ImportError:
    print("⚠️  sympy not available, using basic string matching")
    HAS_SYMPY = False

# Import from MATH folder as fallback
try:
    from envs.MATH.grader import math_equal
    from envs.MATH.verify_utils import grade_answer
    from envs.MATH.parse_utils_qwen import extract_answer as math_extract_answer
except ImportError:
    pass


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


class timeout:
    """Context manager for function timeout (adapted from original.py)"""
    def __init__(self, seconds=5, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{...} content with proper nested brace handling.
    Adapted from test_minerva.py and original.py
    """
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


# Answer normalization constants (from original.py)
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
    "square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours", "km", "units",
    "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents", "degrees", "cm", "gm",
    "pounds", "meters", "meals", "edges", "students", "childrentickets", "multiples",
    "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    Adapted from original.py (Lewkowycz et al. 2022 Appendix D)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold, is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX: \fracab -> \frac{a}{b}, \sqrta -> \sqrt{a}
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def is_equiv(x1: str, x2: str) -> bool:
    """
    Check if two mathematical expressions are equivalent using SymPy.
    Adapted from original.py
    """
    if not HAS_SYMPY:
        # Fallback to string comparison
        return x1.strip() == x2.strip()

    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (sympy.parsing.latex.errors.LaTeXParsingError, sympy.SympifyError, TypeError):
                logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                logger.debug(f"Had trouble simplifying when comparing {x1} and {x2}")
                return False
    except TimeoutError:
        logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except Exception as e:
        logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def extract_answer(answer_str: str) -> str:
    """
    Extract final answer from full response string.
    Priority:
    1. Extract from \\boxed{...}
    2. Fallback to last number/expression
    """
    # Try to extract from \boxed{...}
    boxed = extract_boxed_answer(answer_str)
    if boxed:
        return boxed

    # Fallback: extract last number pattern
    pattern = r"-?\d*\.?\d+"
    matches = re.findall(pattern, answer_str.replace(",", ""))
    if matches:
        return matches[-1]

    return INVALID_ANS


def extract_groundtruth(groundtruth_str: str) -> str:
    """
    Extract ground truth answer from raw groundtruth string.
    For MINERVA, groundtruth is already the answer.
    """
    if groundtruth_str is None:
        return INVALID_ANS

    boxed = extract_boxed_answer(str(groundtruth_str))
    if boxed:
        return boxed

    return str(groundtruth_str).strip()


def verify_answer(
    response: str,
    ground_truth: str,
    use_math_verify: bool = True,
    use_sympy_equiv: bool = True,
) -> dict:
    """
    Verify response against ground truth using multiple methods.
    Returns dict with metrics: math_verify, sympy_equiv, exact_match
    Adapted from original.py and test_minerva.py
    """
    results = {
        "math_verify": False,
        "sympy_equiv": False,
        "exact_match": False,
    }

    # Extract answer from response
    pred = extract_boxed_answer(response)
    if pred is None:
        # Try extracting last number
        pattern = r"-?\d*\.?\d+"
        matches = re.findall(pattern, response.replace(",", ""))
        pred = matches[-1] if matches else None

    if pred is None:
        return results

    # Method 1: math_verify (parse and verify full solution)
    if use_math_verify and HAS_MATH_VERIFY:
        try:
            gold_parsed = parse(
                f"\\boxed{{{ground_truth}}}",
                extraction_config=[LatexExtractionConfig()]
            )
            pred_parsed = parse(
                response,
                extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()]
            )
            results["math_verify"] = bool(verify(gold_parsed, pred_parsed))
        except Exception as e:
            logger.debug(f"math_verify failed: {e}")

    # Method 2: SymPy algebraic equivalence (parse_latex + simplification)
    if use_sympy_equiv and HAS_SYMPY:
        try:
            normalized_pred = normalize_final_answer(pred)
            normalized_truth = normalize_final_answer(ground_truth)
            results["sympy_equiv"] = is_equiv(normalized_pred, normalized_truth)
        except Exception as e:
            logger.debug(f"sympy equivalence check failed: {e}")

    # Method 3: Exact string match (after normalization)
    try:
        normalized_pred = normalize_final_answer(pred)
        normalized_truth = normalize_final_answer(ground_truth)
        results["exact_match"] = (normalized_pred.strip() == normalized_truth.strip())
    except Exception as e:
        logger.debug(f"exact match failed: {e}")

    # If all methods fail, try basic string match as last resort
    if not any(results.values()):
        results["exact_match"] = (pred.strip() == ground_truth.strip())

    return results


def judge_correct(problem_str: str, extracted_groundtruth: Optional[str], answer: str) -> bool:
    """
    Judge if answer is correct using multiple verification methods.
    Returns True if ANY method confirms correctness (math_verify, sympy_equiv, or exact_match).

    This multi-method approach provides robustness:
    - math_verify: checks full solution semantics
    - sympy_equiv: checks algebraic equivalence with normalization
    - exact_match: checks normalized string equality (fallback)
    """
    if answer == INVALID_ANS or extracted_groundtruth == INVALID_ANS:
        return False

    if extracted_groundtruth is None:
        return False

    results = verify_answer(
        answer,
        extracted_groundtruth,
        use_math_verify=True,
        use_sympy_equiv=True
    )

    # Return True if ANY method confirms correctness (majority voting style)
    # Prioritize: math_verify > sympy_equiv > exact_match
    return results.get("math_verify", False) or results.get("sympy_equiv", False) or results.get("exact_match", False)


class Env(CoTEnv):
    """
    MINERVA environment for solving math problems with chain-of-thought.
    """
    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fns,
        rm_call,
        task_desc_str: str = "Please solve the following problem step by step, and put your final answer within \\boxed{}.",
        cot_example_str: str = "",
        problem_format_str: str = "{question}",
        reset=True,
        update_legal_action=True,
    ):
        super().__init__(
            config,
            math_problems,
            llm_gen_fns,
            rm_call,
            task_desc_str if not config.get("cot_prompt", False) else config["cot_prompt"],
            cot_example_str,
            problem_format_str,
            reset,
            sep=config.get("sep", None),
            model_names=config.get("model_names", []),
            update_legal_action=update_legal_action,
        )
        self._stop_str = config.get("stop_str", None)
        self.sep = config.get("sep", None)
        self.model_names = config.get("model_names", [])
        self.double_line_break = config.get("double_line_break", 0)

    def _is_correct(self, completion):
        """Check if completion is correct by extracting answer and judging.
        Uses math_verify library for MINERVA.
        """
        extracted_answer = extract_answer(completion)
        return judge_correct(self.math_problem["question"], self.math_problem["answer"], extracted_answer)

    def get_reward(self):
        """Get reward from reward model.
        For MINERVA, returns 0 as placeholder (actual reward comes from rm_call).
        """
        return 0

    def post_process_act(self, action: str):
        """Post-process action for MINERVA task."""
        action = action.strip()

        if self.direct_io == 2:
            return action
        elif self.direct_io == 1:
            # For best-of-n: return action as-is
            return action.strip()
        else:
            # For CoT: return action as-is
            return action
