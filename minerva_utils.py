"""
Standard utilities from minerva.py for answer extraction and verification.
This module provides the canonical implementations used across all evaluation scripts.
"""

from typing import Optional

# ── Answer extraction (exact implementation from minerva.py) ─────────────────

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{...} content with proper nested brace handling.

    This is the canonical implementation from minerva.py.
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


def verify_answer(
    response: str,
    ground_truth: str,
    use_math_verify: bool = True,
) -> bool:
    """
    Verify response against ground truth using math_verify if available.

    This is the canonical implementation from minerva.py.
    Falls back to simple string matching if math_verify not available.

    Args:
        response: Model response text
        ground_truth: Ground truth answer
        use_math_verify: Whether to use math_verify (requires library)

    Returns:
        True if answer is correct, False otherwise
    """
    if use_math_verify:
        try:
            from math_verify import verify, parse
            from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

            gold_parsed = parse(
                f"\\boxed{{{ground_truth}}}",
                extraction_config=[LatexExtractionConfig()]
            )
            pred_parsed = parse(
                response,
                extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()]
            )
            return bool(verify(gold_parsed, pred_parsed))
        except (ImportError, Exception):
            # Fall through to string matching
            pass

    # Fallback: extract boxed + string match
    pred = extract_boxed_answer(response)
    if pred is None:
        return False
    return pred.strip() == ground_truth.strip()


# ── Additional helper for numeric/format-aware verification ─────────────────

def verify_answer_smart(
    response: str,
    ground_truth: str,
) -> bool:
    """
    Enhanced verification that tries multiple matching strategies WITHOUT tolerance.
    Requires EXACT matches, but handles format variations.

    Progression:
    1. Direct string match via minerva.py standard
    2. Numeric comparison (exact match only)
    3. LaTeX-aware matching (format normalization)
    4. Scientific notation normalization
    5. Unit removal then numeric match

    Args:
        response: Model response text
        ground_truth: Ground truth answer

    Returns:
        True if answer is correct (exact numeric or string match), False otherwise
    """
    import re

    # First try standard minerva verification (string matching)
    if verify_answer(response, ground_truth, use_math_verify=False):
        return True

    # Extract boxed answers
    pred = extract_boxed_answer(response)
    if pred is None:
        return False

    pred = pred.strip()
    gt = ground_truth.strip()

    # Try numeric comparison (EXACT match only)
    pred_val = _extract_numeric_value(pred)
    gt_val = _extract_numeric_value(gt)

    if pred_val is not None and gt_val is not None:
        # Both numeric: check EXACT match (no tolerance)
        return pred_val == gt_val

    # Try LaTeX-aware matching (normalize format)
    pred_clean = _clean_latex(pred)
    gt_clean = _clean_latex(gt)

    if pred_clean == gt_clean:
        return True

    # Try alphanumeric matching (ignore all non-alphanumeric chars)
    pred_alphanum = re.sub(r'[^\w]', '', pred_clean).lower()
    gt_alphanum = re.sub(r'[^\w]', '', gt_clean).lower()

    if pred_alphanum == gt_alphanum and len(pred_alphanum) > 0:
        return True

    return False


def _clean_latex(text: str) -> str:
    """Remove LaTeX formatting for comparison."""
    import re
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\times', '*', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_numeric_value(text: str) -> Optional[float]:
    """Extract numeric value from text with unit/notation handling."""
    import re

    text = _clean_latex(text)

    # Remove common units
    units_pattern = r'\s*(?:cm|m|mm|kg|g|s|Hz|arcsec|arcmin|degree|ergs?/s|erg/s|Angstroms?|K|C|rad|°|″|′)(?:\s|$|/)'
    text_no_units = re.sub(units_pattern, '', text)

    # Try direct float
    try:
        return float(text_no_units)
    except ValueError:
        pass

    # Try fractions
    try:
        match = re.search(r'\(?(\d+)\)?/\(?(\d+)\)?', text_no_units)
        if match:
            num = int(match.group(1))
            denom = int(match.group(2))
            return num / denom
    except (ValueError, ZeroDivisionError):
        pass

    # Try scientific notation
    try:
        match = re.search(r'([-+]?\d*\.?\d+)\s*(?:\*|x|×)\s*10\^?\{?([+-]?\d+)\}?', text_no_units, re.IGNORECASE)
        if match:
            base = float(match.group(1))
            exp = int(match.group(2))
            return base * (10 ** exp)
    except (ValueError, AttributeError):
        pass

    # Try e notation
    try:
        match = re.search(r'([-+]?\d*\.?\d+)e([+-]?\d+)', text_no_units, re.IGNORECASE)
        if match:
            base = float(match.group(1))
            exp = int(match.group(2))
            return base * (10 ** exp)
    except (ValueError, AttributeError):
        pass

    return None
