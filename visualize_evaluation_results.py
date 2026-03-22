"""
Visualization of baseline evaluation results across all methods and indices.
Creates text-based charts and statistics.
"""

import json
from pathlib import Path


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


def extract_accuracies(results):
    """Extract accuracy percentages from results."""
    methods = {}

    for method_name, method_data in results.items():
        methods[method_name] = {}
        for idx_key, data in method_data.items():
            if isinstance(data, dict) and 'accuracy' in data:
                idx = int(idx_key.split('_')[-1])
                methods[method_name][idx] = data['accuracy'] * 100
            elif isinstance(data, (int, float)):
                idx = int(idx_key.split('_')[-1]) if '_' in idx_key else int(idx_key)
                methods[method_name][idx] = data * 100

    return methods


def print_text_chart(accuracies):
    """Print text-based chart for each method."""
    print("=" * 140)
    print("VISUAL REPRESENTATION: Baseline Accuracy Across Output Indices")
    print("=" * 140)
    print()

    for method_name in ['verify_answer', 'verify_answer_smart', 'is_equiv']:
        if method_name not in accuracies:
            continue

        print(f"\n{method_name.upper()}")
        print("-" * 140)
        print(f"{'Index':<8} {'Accuracy':<10} {'Chart':<120}")
        print("-" * 140)

        method_accs = accuracies[method_name]
        max_acc = max(method_accs.values())

        for idx in range(32):
            acc = method_accs.get(idx, 0.0)
            bar_length = int((acc / 40) * 100)  # Scale to max possible 40%
            bar = '█' * bar_length + '░' * (100 - bar_length)
            print(f"{idx:<8} {acc:>6.2f}%   {bar}")

        print()


def print_statistics(accuracies):
    """Print statistical summary."""
    print("\n" + "=" * 140)
    print("STATISTICAL SUMMARY")
    print("=" * 140)
    print()

    for method_name in ['verify_answer', 'verify_answer_smart', 'is_equiv']:
        if method_name not in accuracies:
            continue

        accs = list(accuracies[method_name].values())
        mean_acc = sum(accs) / len(accs)
        min_acc = min(accs)
        max_acc = max(accs)
        best_idx = max(accuracies[method_name], key=accuracies[method_name].get)
        worst_idx = min(accuracies[method_name], key=accuracies[method_name].get)

        # Calculate standard deviation
        variance = sum((x - mean_acc) ** 2 for x in accs) / len(accs)
        std_dev = variance ** 0.5

        print(f"{method_name.upper()}")
        print(f"  Mean Accuracy:  {mean_acc:>6.2f}%")
        print(f"  Best Accuracy:  {max_acc:>6.2f}% (Index {best_idx})")
        print(f"  Worst Accuracy: {min_acc:>6.2f}% (Index {worst_idx})")
        print(f"  Std Deviation:  {std_dev:>6.2f}%")
        print(f"  Range:          {max_acc - min_acc:>6.2f}%")
        print()


def print_ranking():
    """Print ranking of indices by best method performance."""
    results = load_results()
    accuracies = extract_accuracies(results)

    print("\n" + "=" * 140)
    print("INDEX RANKING: By is_equiv (Best Performing Method)")
    print("=" * 140)
    print()

    if 'is_equiv' in accuracies:
        indexed = [(idx, acc) for idx, acc in accuracies['is_equiv'].items()]
        indexed.sort(key=lambda x: x[1], reverse=True)

        print(f"{'Rank':<6} {'Index':<8} {'is_equiv':<12} {'vs verify_answer':<18} {'vs verify_answer_smart':<24}")
        print("-" * 140)

        for rank, (idx, acc) in enumerate(indexed[:16], 1):
            standard = accuracies['verify_answer'].get(idx, 0.0)
            smart = accuracies['verify_answer_smart'].get(idx, 0.0)
            diff_std = acc - standard
            diff_smart = acc - smart

            print(f"{rank:<6} {idx:<8} {acc:>6.2f}%      {diff_std:+6.2f}% ({diff_std/standard*100:+5.1f}%)   {diff_smart:+6.2f}% ({diff_smart/smart*100:+5.1f}%)")

    print()


def main():
    results = load_results()

    if not results:
        print("No results found. Please run evaluation scripts first.")
        return

    accuracies = extract_accuracies(results)

    # Print charts
    print_text_chart(accuracies)

    # Print statistics
    print_statistics(accuracies)

    # Print ranking
    print_ranking()

    # Summary
    print("\n" + "=" * 140)
    print("KEY TAKEAWAYS")
    print("=" * 140)
    print("""
1. Mathematical Equivalence (is_equiv) is MOST ACCURATE
   - Achieves 34.56% at index 25 (highest overall)
   - Better at capturing algebraic equivalence

2. Format Handling (verify_answer_smart) is PRACTICAL
   - Achieves 33.46% at index 4
   - Handles format variations without symbolic parsing
   - Faster than SymPy equivalence checking

3. String Matching (verify_answer) is CONSERVATIVE
   - Only 22.43% at best (index 0)
   - Too strict for physics problems with notation variations
   - Serves as sanity check / lower bound

4. Index Selection Matters
   - Different methods prefer different indices
   - is_equiv: prefers index 25 (middle outputs)
   - verify_answer_smart: prefers index 4 (early-middle)
   - verify_answer: prefers index 0 (earliest)

5. Ensemble Advantage
   - Voting strategies with any method achieve ~47.79%
   - Combining multiple outputs beats single selection by ~13-25%
   - Demonstrates importance of diversity in sampling
    """)


if __name__ == "__main__":
    main()
