"""
Analyze MINERVA evaluation results to understand differences between
majority voting and PRM last max strategies.
"""

import json
from pathlib import Path
from typing import Dict, List

def analyze_results():
    with open('/Users/luungoc/Project/compute-optimal-tts/evaluation_results.json', 'r') as f:
        results = json.load(f)

    detailed = results['detailed_results']

    # Categorize results
    both_correct = [r for r in detailed if r['mv_correct'] and r['prm_correct']]
    both_wrong = [r for r in detailed if not r['mv_correct'] and not r['prm_correct']]
    mv_better = [r for r in detailed if r['mv_correct'] and not r['prm_correct']]
    prm_better = [r for r in detailed if r['prm_correct'] and not r['mv_correct']]

    print("=" * 80)
    print("MINERVA EVALUATION ANALYSIS")
    print("=" * 80)
    print()

    # Summary statistics
    print("SUMMARY STATISTICS")
    print("-" * 80)
    print(f"Total questions: {len(detailed)}")
    print(f"Both correct:    {len(both_correct):3d} ({len(both_correct)/len(detailed)*100:5.2f}%)")
    print(f"Both wrong:      {len(both_wrong):3d} ({len(both_wrong)/len(detailed)*100:5.2f}%)")
    print(f"MV better:       {len(mv_better):3d} ({len(mv_better)/len(detailed)*100:5.2f}%)")
    print(f"PRM better:      {len(prm_better):3d} ({len(prm_better)/len(detailed)*100:5.2f}%)")
    print()

    print("ACCURACY BREAKDOWN")
    print("-" * 80)
    print(f"Majority Vote:   {results['summary']['majority_vote_correct']:3d}/{len(detailed)} = {results['summary']['majority_vote_accuracy']:5.2f}%")
    print(f"PRM Last Max:    {results['summary']['prm_last_max_correct']:3d}/{len(detailed)} = {results['summary']['prm_last_max_accuracy']:5.2f}%")
    print(f"Difference:      {len(mv_better) - len(prm_better):3d} (MV advantage)")
    print()

    # Analyze cases where MV is better
    print("=" * 80)
    print("CASES WHERE MAJORITY VOTE OUTPERFORMS PRM LAST MAX")
    print(f"Count: {len(mv_better)}")
    print("=" * 80)
    print()

    if mv_better:
        # Group by answer type patterns
        patterns = {
            'numeric_diff': [],  # Same numeric format but different values
            'format_diff': [],   # Different formats (e.g., scientific vs decimal)
            'units': [],         # Different units
            'precision': [],     # Different precision (e.g., 0.022 vs 0.0217)
        }

        for case in mv_better:
            mv_ans = case['mv_answer']
            prm_ans = case['prm_answer']

            if not mv_ans or not prm_ans:
                continue

            # Check if it's a precision/rounding issue
            if '.' in str(mv_ans) and '.' in str(prm_ans):
                mv_parts = str(mv_ans).split('.')
                prm_parts = str(prm_ans).split('.')
                if len(mv_parts[1]) != len(prm_parts[1]):
                    patterns['precision'].append(case)
                    continue

            # Check for units
            if any(unit in str(prm_ans) for unit in ['cm', 'm', 'erg', 'K', 'Hz']):
                patterns['units'].append(case)
                continue

            # Check for format differences
            if '\\' in str(prm_ans) or 'e' in str(prm_ans).lower():
                patterns['format_diff'].append(case)
                continue

            patterns['numeric_diff'].append(case)

        for pattern_name, cases in patterns.items():
            if cases:
                print(f"\n{pattern_name.upper()} ({len(cases)} cases):")
                print("-" * 80)
                for i, case in enumerate(cases[:3]):
                    print(f"  GT:  {case['ground_truth']}")
                    print(f"  MV:  {case['mv_answer']} ✓")
                    print(f"  PRM: {case['prm_answer']} ✗")
                    print()

    # Analyze cases where PRM is better
    print("=" * 80)
    print("CASES WHERE PRM LAST MAX OUTPERFORMS MAJORITY VOTE")
    print(f"Count: {len(prm_better)}")
    print("=" * 80)
    print()

    if prm_better:
        print("Examples:")
        print("-" * 80)
        for i, case in enumerate(prm_better[:5]):
            print(f"{i+1}. GT:  {case['ground_truth']}")
            print(f"   MV:  {case['mv_answer']} ✗")
            print(f"   PRM: {case['prm_answer']} ✓")
            print()

    # Show cases where both are wrong
    print("=" * 80)
    print("CASES WHERE BOTH STRATEGIES FAIL")
    print(f"Count: {len(both_wrong)}")
    print("=" * 80)
    print()

    if both_wrong:
        print("Sample failed questions:")
        print("-" * 80)
        for i, case in enumerate(both_wrong[:5]):
            print(f"{i+1}. GT:  {case['ground_truth']}")
            print(f"   MV:  {case['mv_answer']}")
            print(f"   PRM: {case['prm_answer']}")
            print()

    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print(f"""
1. Majority Vote significantly outperforms PRM Last Max
   - MV wins by {len(mv_better)} questions
   - PRM wins by {len(prm_better)} questions

2. Primary reason for MV superiority:
   - MV aggregates multiple answers, reducing outliers
   - PRM Last Max relies on a single "best" output

3. Recommendation:
   - Use Majority Vote as the primary strategy
   - Could potentially combine both for further improvement
   - Consider ensemble methods with confidence weighting
""")


if __name__ == "__main__":
    analyze_results()
