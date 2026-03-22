"""
MINERVA task evaluation module using math_verify or fallback to MATH folder functions.
Adapted from test_minerva.py
"""

import re
from typing import Optional
from envs.base_env import CoTEnv, NoLegalActionException, INVALID_ANS

# Try to import math_verify, fallback to MATH folder functions
try:
    from math_verify import verify, parse
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    HAS_MATH_VERIFY = True
except ImportError:
    print("⚠️  math_verify not available, using MATH folder functions as fallback")
    HAS_MATH_VERIFY = False
    # Import from MATH folder
    from envs.MATH.grader import math_equal
    from envs.MATH.verify_utils import grade_answer
    from envs.MATH.parse_utils_qwen import extract_answer as math_extract_answer


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{...} content with proper nested brace handling.
    Adapted from test_minerva.py
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
) -> bool:
    """
    Verify response against ground truth.
    Uses math_verify if available, otherwise falls back to MATH folder functions.
    Adapted from test_minerva.py
    """
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
            return bool(verify(gold_parsed, pred_parsed))
        except Exception:
            # Fallback: try MATH folder functions
            pass

    # Fallback 1: Use MATH folder functions if math_verify not available
    if not HAS_MATH_VERIFY:
        # Extract answer from response
        pred = extract_boxed_answer(response)
        if pred is None:
            # Try extracting last number
            pattern = r"-?\d*\.?\d+"
            matches = re.findall(pattern, response.replace(",", ""))
            pred = matches[-1] if matches else None

        if pred is None:
            return False

        try:
            # Use MATH folder's math_equal
            return math_equal(pred.strip(), ground_truth.strip(), include_percentage=True, is_close=True)
        except Exception:
            # Last resort: string match
            return pred.strip() == ground_truth.strip()

    # Fallback 2: extract boxed + string match (if math_verify method failed)
    pred = extract_boxed_answer(response)
    if pred is None:
        return False
    return pred.strip() == ground_truth.strip()


def judge_correct(problem_str: str, extracted_groundtruth: Optional[str], answer: str) -> bool:
    """
    Judge if answer is correct using math_verify.
    """
    if answer == INVALID_ANS or extracted_groundtruth == INVALID_ANS:
        return False

    if extracted_groundtruth is None:
        return False

    return verify_answer(answer, extracted_groundtruth, use_math_verify=True)


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
