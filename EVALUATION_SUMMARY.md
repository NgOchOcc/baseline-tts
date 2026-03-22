# MINERVA Questions Evaluation Report

## Overview

Evaluated **272 MINERVA physics questions** using two distinct strategies:
1. **Majority Voting (MV)**: Aggregate answers from all outputs via voting
2. **PRM Last Max (PRM)**: Select output with highest final reward score

## Results

### Accuracy Comparison

| Strategy | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| **Majority Vote** | **136** | 272 | **50.00%** |
| **PRM Last Max** | 117 | 272 | 43.01% |
| **Difference** | **+19** | — | **+6.99%** |

### Performance Breakdown

| Category | Count | Percentage |
|----------|-------|-----------|
| Both correct | 111 | 40.81% |
| Both wrong | 130 | 47.79% |
| MV better (MV ✓, PRM ✗) | 25 | 9.19% |
| PRM better (PRM ✓, MV ✗) | 6 | 2.21% |

## Key Findings

### Why Majority Vote Wins

**Majority Vote Superiority: 25 vs 6**

Majority voting outperforms PRM Last Max in 25 cases across these patterns:

1. **Format Differences (4 cases)**
   - Symbolic expressions handled better with MV
   - Example: MV correctly handles `\sqrt{4\pi G \rho_0 r_0^2}`

2. **Numeric Errors (10 cases)**
   - PRM selects outputs with wrong values
   - Example: Ground truth `10`, MV `10` ✓, PRM `40` ✗

3. **Unit Handling (9 cases)**
   - PRM often includes/misinterprets units
   - Example: GT `9.6`, MV `9.7`, PRM `97 Angstroms`

4. **Precision Issues (2 cases)**
   - PRM occasionally selects wrong order of magnitude
   - Example: GT `8.7e8`, MV `8.8e8`, PRM `8.68e10`

### Why PRM Sometimes Wins (6 cases)

PRM's reward-based selection helps when:
- Majority vote is split with outliers
- Single correct output significantly outperforms others
- Example: GT `1.75`, MV `3.08`, PRM `1.75` ✓

### Common Failure Cases (130 questions)

Both strategies fail primarily on:

1. **Symbolic/Formula Expressions**
   - Unable to parse complex mathematical expressions
   - Example: `\frac{2\pi c^2 R^2}{d^2\lambda^5[e^{hc/(\lambda kT)}-1]}`

2. **Implicit Answer Transformations**
   - Ground truth: `np.arcsin(10/13)` vs Model: `\arcsin(1/1.3)`
   - Requires symbolic math to verify equivalence

3. **Unit/Scale Confusion**
   - Model adds units not in ground truth
   - Model confuses unit scales (km/s vs m/s)

## Implementation Details

### Smart Answer Verification

The evaluation uses intelligent comparison:

```python
# 1. Direct string match
"1.57" == "1.57" → True

# 2. Numeric comparison with tolerance
1.57 ≈ 1.6 (within 2% tolerance) → True

# 3. Scientific notation handling
"4.5e33" == "4.5 × 10^33" → True

# 4. Unit removal & parsing
"1.75 mL" → parse 1.75 → compare
"6 × 10^{-3}" → 0.006 → compare

# 5. LaTeX cleaning & matching
"\sqrt{4\pi...}" → cleaned format matching
```

### Tolerance Settings

- **Default tolerance**: 2% relative error
- **Small numbers** (<0.1): Same tolerance but based on magnitude
- **Very small numbers** (<1e-10): Absolute tolerance 1e-5

## Recommendations

### For Physics QA Systems

1. **Primary Strategy**: Use Majority Voting
   - Better reliability through aggregation
   - Reduces impact of model hallucinations
   - Recommended for production use

2. **Ensemble Approach**: Combine both strategies
   ```
   - If both agree → High confidence
   - If only MV correct → Use MV (75% of disagreement cases)
   - If only PRM correct → Use PRM (25% of disagreement cases)
   - If both wrong → Requires better model/dataset
   ```

3. **Improve Symbolic Expression Handling**
   - Current 50% accuracy limited by symbolic math
   - Consider SymPy or symbolic reasoning for improvement
   - Would significantly boost both strategies

### For PRM Optimization

PRM Last Max underperforms because:
- Single output selection is fragile
- Reward scores don't perfectly correlate with correctness
- Consider: Average reward across steps, or use PRM threshold

Potential improvements:
- Use PRM ranking (top-3) then apply MV
- Weight MV votes by PRM scores
- Hybrid: Use PRM confidence threshold

## Files Generated

- `evaluate_minerva_questions.py` - Main evaluation script
- `analyze_evaluation_results.py` - Detailed analysis script
- `evaluation_results.json` - Complete results with per-question data

## Running the Evaluation

```bash
# Run evaluation
python evaluate_minerva_questions.py

# Analyze results
python analyze_evaluation_results.py

# View results
cat evaluation_results.json | python -m json.tool | head -100
```

## Dataset Characteristics

- **Total Questions**: 272 from MINERVA physics dataset
- **Answer Types**:
  - Numeric (decimal, scientific notation)
  - Symbolic (formulas, expressions)
  - Formatted text (with units)
- **Model**: Qwen2.5-7B-Instruct with PRM scoring
- **Samples per Question**: ~4-7 outputs per question

## Conclusion

**Majority voting** provides a more robust and reliable approach for evaluating MINERVA question answering, with **50% accuracy** vs **43% for PRM Last Max**. The key advantage is reducing outliers through aggregation. For production systems, MV should be the primary strategy, potentially enhanced with ensemble voting and better symbolic expression handling.
