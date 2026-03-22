# MINERVA Standard Utilities Usage Guide

## Overview

This document describes the standard utilities for answer extraction and verification in MINERVA evaluation, following the conventions from `minerva.py`.

## Standard Functions

### 1. `extract_boxed_answer(text: str) -> Optional[str]`

**Purpose:** Extract the last `\boxed{...}` content with proper nested brace handling.

**Implementation:** From `minerva.py:50-70`

```python
from minerva_utils import extract_boxed_answer

response = "The answer is $x=2$. Therefore \\boxed{2}"
answer = extract_boxed_answer(response)
# Returns: "2"
```

**Features:**
- ✅ Handles nested braces correctly
- ✅ Extracts **last** boxed answer if multiple exist
- ✅ Returns None if no boxed answer found
- ✅ Strips whitespace from result

### 2. `verify_answer(response: str, ground_truth: str, use_math_verify: bool = False) -> bool`

**Purpose:** Verify response against ground truth using `math_verify` if available, else fallback to string matching.

**Implementation:** From `minerva.py:73-93`

```python
from minerva_utils import verify_answer

# String matching fallback (when math_verify unavailable)
is_correct = verify_answer(
    response="\\boxed{2}",
    ground_truth="2",
    use_math_verify=False
)
# Returns: True
```

**Behavior:**
- If `use_math_verify=True` and library available: Uses symbolic math verification
- If `use_math_verify=False` or math_verify unavailable:
  - Extracts boxed answer from response
  - Compares directly with ground truth (case-sensitive string match)
  - Returns True only if exact match

**Requirements:**
- Response should contain `\boxed{...}` format
- Ground truth should be plain text/number

### 3. `verify_answer_smart(response: str, ground_truth: str, tolerance: float = 0.02) -> bool`

**Purpose:** Enhanced verification combining standard approach with numeric/format handling.

**Implementation:** Custom extension in `minerva_utils.py`

```python
from minerva_utils import verify_answer_smart

# Numeric comparison with tolerance
is_correct = verify_answer_smart(
    response="\\boxed{1.57}",
    ground_truth="1.6",
    tolerance=0.02  # 2% relative error
)
# Returns: True (1.57 within 2% of 1.6)

# Scientific notation handling
is_correct = verify_answer_smart(
    response="\\boxed{4.5 \\times 10^{33}}",
    ground_truth="4.5e33",
    tolerance=0.02
)
# Returns: True

# Unit handling
is_correct = verify_answer_smart(
    response="\\boxed{1.75 \\text{ mL}}",
    ground_truth="1.75"
)
# Returns: True
```

**Verification Strategy (in order):**
1. Standard string matching (minerva.py)
2. Numeric comparison with tolerance
3. LaTeX format cleaning and comparison
4. Alphanumeric matching (ignoring special characters)

**Parameters:**
- `response`: Model response with `\boxed{...}`
- `ground_truth`: Ground truth answer (plain format)
- `tolerance`: Relative error threshold (default 2%)

**Features:**
- ✅ Numeric comparison with adaptive tolerance
- ✅ Scientific notation (4.5e33 vs 4.5×10³³)
- ✅ Fraction handling (1/20 vs 0.05)
- ✅ Unit removal (1.75 mL → 1.75)
- ✅ LaTeX formatting normalization
- ✅ Alphanumeric matching fallback

## Usage Examples

### Example 1: Standard Verification (minerva.py)

```python
from minerva_utils import extract_boxed_answer, verify_answer

# Extract answer
response = "Solving: x + 2 = 4, so x = 2. \\boxed{2}"
answer = extract_boxed_answer(response)  # "2"

# Verify with ground truth
is_correct = verify_answer(
    f"\\boxed{{{answer}}}",
    ground_truth="2",
    use_math_verify=False
)  # True
```

### Example 2: Smart Verification for Physics

```python
from minerva_utils import verify_answer_smart

# Case 1: Numeric precision
response = "The distance is \\boxed{9.7}"
is_correct = verify_answer_smart(response, "9.6", tolerance=0.02)  # True

# Case 2: Scientific notation
response = "Energy = \\boxed{4.5 \\times 10^{33} \\text{ erg/s}}"
is_correct = verify_answer_smart(response, "4.5e33")  # True

# Case 3: Complex expressions
response = "Result: \\boxed{\\arcsin(1.3 \\sin(\\theta_w))}"
is_correct = verify_answer_smart(
    response,
    "\\arcsin{1.3 \\sin{\\theta_w}}",
    tolerance=0.02
)  # True
```

### Example 3: Voting Evaluation

```python
from minerva_utils import verify_answer_smart

# Apply voting strategy
answers = ["2", "2", "3", "2"]  # Multiple model outputs
from collections import Counter
best_answer = Counter(answers).most_common(1)[0][0]

# Verify result
is_correct = verify_answer_smart(
    f"\\boxed{{{best_answer}}}",
    ground_truth="2"
)
```

## API Reference

### `extract_boxed_answer(text: str) -> Optional[str]`

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | str | Response text containing `\boxed{...}` |
| **Returns** | str/None | Last boxed content or None |

### `verify_answer(...) -> bool`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `response` | str | - | Response with `\boxed{...}` |
| `ground_truth` | str | - | Ground truth answer |
| `use_math_verify` | bool | False | Enable symbolic math |
| **Returns** | bool | - | True if correct |

### `verify_answer_smart(...) -> bool`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `response` | str | - | Response with `\boxed{...}` |
| `ground_truth` | str | - | Ground truth answer |
| `tolerance` | float | 0.02 | Relative error tolerance |
| **Returns** | bool | - | True if correct |

## When to Use Which

| Scenario | Function |
|----------|----------|
| Extract last `\boxed{}` | `extract_boxed_answer()` |
| Exact string comparison | `verify_answer(..., use_math_verify=False)` |
| Symbolic math verification | `verify_answer(..., use_math_verify=True)` |
| Numeric with tolerance | `verify_answer_smart()` |
| Production system | `verify_answer_smart()` |
| Research baseline | `verify_answer(..., use_math_verify=False)` |

## Accuracy Implications

### Standard String Matching Only
```
Result: 47.79% (130/272 for majority vote)
- Fails on: Numeric precision, unit variations, notation differences
```

### With Smart Verification
```
Result: ~50% (with numeric tolerance)
- Gains: Handles scientific notation, units, precision differences
```

### Full Symbolic Verification (requires math_verify)
```
Result: ~55%+ (theoretical, not tested)
- Gains: Symbolic expression equivalence
```

## Implementation Notes

1. **Canonical vs Extended:**
   - `verify_answer` = canonical implementation from minerva.py
   - `verify_answer_smart` = extended version with numeric handling

2. **String Matching Limitation:**
   - minerva.py standard uses string matching when `math_verify` unavailable
   - This is strict: `1.57 ≠ 1.6` even though numerically close
   - `verify_answer_smart` adds tolerance for practical use

3. **Consistency:**
   - All evaluation scripts use `extract_boxed_answer` (standard)
   - Scripts use `verify_answer_smart` for evaluation
   - But comparison with minerva.py baseline uses `verify_answer`

## Files

- `minerva_utils.py` - Standard utilities with canonical implementations
- `minerva_voting_evaluation.py` - Voting evaluation using standards
- `compare_voting_strategies.py` - Strategy comparison
- `evaluate_minerva_questions.py` - Initial analysis
- `minerva.py` - Original reference implementation

## References

- Standard implementations: `minerva.py:50-93`
- Voting strategies: `vote_utils.py`
- Extraction: `extract_boxed_answer` - handles nested braces correctly
- Verification: `verify_answer` - string matching with math_verify option
