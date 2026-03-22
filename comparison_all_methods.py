"""
Comprehensive comparison of all evaluation methods for baseline accuracy across output indices.
Compares:
1. verify_answer (standard string matching from minerva.py)
2. verify_answer_smart (exact numeric + format normalization)
3. is_equiv (SymPy-based mathematical equivalence)
"""

import json
from pathlib import Path
from typing import Dict, List

def load_results():
    """Load all evaluation results."""
    base_dir = Path("/Users/luungoc/Project/compute-optimal-tts/MINERVA_best_of_n")
    base_dir = base_dir / "Qwen2.5-7B-Instruct" / "Qwen2.5-Math-PRM-7B" / "seed_0_width_32_num_seq_32_num_q_0"

    results = {}

    # Load standard verify_answer results
    file_standard = base_dir / "baseline_result_all_indices.json"
    if file_standard.exists():
        with open(file_standard, 'r') as f:
            data = json.load(f)
            results['verify_answer'] = data['summary']

    # Load smart verify_answer_smart results
    file_smart = base_dir / "baseline_result_all_indices_smart.json"
    if file_smart.exists():
        with open(file_smart, 'r') as f:
            data = json.load(f)
            results['verify_answer_smart'] = data['summary']

    # Load is_equiv (SymPy) results
    file_github = base_dir / "baseline_result_all_indices_github_methods.json"
    if file_github.exists():
        with open(file_github, 'r') as f:
            data = json.load(f)
            results['is_equiv'] = data['is_equiv_summary']

    return results


def print_comparison():
    """Print comprehensive comparison table."""
    results = load_results()

    print("=" * 150)
    print("COMPREHENSIVE COMPARISON: Baseline Accuracy for Each Output Index (0-31)")
    print("=" * 150)
    print()

    if not results:
        print("No results found. Please run evaluation scripts first.")
        return

    # Extract accuracy for each method and index
    methods = list(results.keys())
    accuracies = {}

    for method in methods:
        accuracies[method] = {}
        for idx_key, data in results[method].items():
            if isinstance(data, dict) and 'accuracy' in data:
                idx = int(idx_key.split('_')[-1])
                accuracies[method][idx] = data['accuracy'] * 100  # Convert to percentage
            elif isinstance(data, (int, float)):
                # Handle direct accuracy values
                idx = int(idx_key.split('_')[-1]) if '_' in idx_key else int(idx_key)
                accuracies[method][idx] = data * 100

    # Create comparison table
    print(f"{'Index':<8} ", end="")
    for method in methods:
        print(f"{method:<20} ", end="")
    print()
    print("-" * 150)

    # Print for each index
    for idx in range(32):
        print(f"{idx:<8} ", end="")
        for method in methods:
            acc = accuracies[method].get(idx, 0.0)
            print(f"{acc:>6.2f}%         ", end="")
        print()

    print()
    print("=" * 150)
    print("BEST PERFORMING INDEX FOR EACH METHOD")
    print("=" * 150)
    print()

    # Find best index for each method
    for method in methods:
        best_idx = max(accuracies[method], key=accuracies[method].get)
        best_acc = accuracies[method][best_idx]
        print(f"{method:<20} → Index {best_idx:<3} with {best_acc:>6.2f}% accuracy")

    print()
    print("=" * 150)
    print("METHOD COMPARISON AT BEST INDICES")
    print("=" * 150)
    print()

    # Compare all methods at their best indices
    comparison_data = []
    for method in methods:
        best_idx = max(accuracies[method], key=accuracies[method].get)
        best_acc = accuracies[method][best_idx]
        comparison_data.append((method, best_idx, best_acc))

    comparison_data.sort(key=lambda x: x[2], reverse=True)

    print(f"{'Method':<25} {'Best Index':<15} {'Best Accuracy':<15}")
    print("-" * 150)
    for method, best_idx, best_acc in comparison_data:
        print(f"{method:<25} {best_idx:<15} {best_acc:>6.2f}%")

    print()
    print("=" * 150)
    print("INTERPRETATION")
    print("=" * 150)
    print("""
1. verify_answer (Standard String Matching):
   - Canonical method from minerva.py
   - Requires exact boxed answer match as string
   - Best performance on early outputs
   - Baseline for other methods

2. verify_answer_smart (Exact Numeric + Format Normalization):
   - Handles different formats (scientific notation, fractions, units)
   - Requires EXACT numeric matches (no tolerance)
   - Better at index 4 (33.46%) vs index 0 (33.09%)
   - Improvement over string matching through format flexibility

3. is_equiv (SymPy-based Mathematical Equivalence):
   - Uses SymPy to parse LaTeX and simplify mathematically
   - Checks if difference between expressions equals 0
   - Best at index 25 (34.56%)
   - Highest overall accuracy across all indices
   - Handles algebraic equivalence (e.g., 1/2 = 0.5)

KEY INSIGHT: is_equiv gives best results (34.56%), suggesting that
mathematical equivalence checking (not just string matching) is important
for accurate evaluation of MINERVA math questions.
    """)


if __name__ == "__main__":
    print_comparison()
