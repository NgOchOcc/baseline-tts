"""
Baseline evaluation testing ALL output indices (0-31).
For each output index, calculate accuracy separately using output[idx] as final answer.
Uses verify_answer() with standard string matching.
"""

import json
from pathlib import Path
from typing import Dict, List
from minerva_utils import verify_answer


def evaluate_all_indices():
    """Evaluate accuracy for each output index (0-31)."""
    base_dir = Path("/Users/luungoc/Project/compute-optimal-tts/MINERVA_best_of_n")
    base_dir = base_dir / "Qwen2.5-7B-Instruct" / "Qwen2.5-Math-PRM-7B" / "seed_0_width_32_num_seq_32_num_q_0"

    # Get all question directories
    question_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("question_")],
        key=lambda x: int(x.name.split("_")[1])
    )

    print(f"Found {len(question_dirs)} question directories")
    print("=" * 100)
    print("BASELINE EVALUATION: Testing each output index (0-31)")
    print("=" * 100)
    print()

    # Track results for each output index
    # results[idx] = {'correct': count, 'total': count}
    index_results = {idx: {'correct': 0, 'total': 0} for idx in range(32)}
    detailed_results_per_index = {idx: [] for idx in range(32)}

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
        for output_idx in range(min(32, len(outputs))):  # Cap at 32 indices
            try:
                output_item = outputs[output_idx]
                baseline_answer = output_item.get('extracted_answer')

                # Verify answer using standard verify_answer
                if baseline_answer:
                    is_correct = verify_answer(
                        f"\\boxed{{{baseline_answer}}}",
                        ground_truth,
                        use_math_verify=True
                    )
                else:
                    is_correct = False

                index_results[output_idx]['total'] += 1
                if is_correct:
                    index_results[output_idx]['correct'] += 1

                # Store detailed result
                detailed_results_per_index[output_idx].append({
                    'question_idx': q_idx,
                    'question_dir': question_dir.name,
                    'ground_truth': ground_truth,
                    'answer': baseline_answer,
                    'is_correct': is_correct,
                })

            except (IndexError, KeyError, TypeError):
                pass

        # Print progress
        if (q_idx + 1) % 50 == 0 or (q_idx + 1) == len(question_dirs):
            print(f"[{q_idx+1}/{len(question_dirs)}] Processed")

    # Print summary table
    print("\n" + "=" * 100)
    print("RESULTS: Accuracy for each output index (0-31)")
    print("=" * 100)
    print()
    print(f"{'Index':<8} {'Correct':<10} {'Total':<10} {'Accuracy':<12} {'Status'}")
    print("-" * 100)

    # Calculate and sort results
    results_with_acc = []
    for idx in range(32):
        total = index_results[idx]['total']
        correct = index_results[idx]['correct']
        accuracy = (correct / total * 100) if total > 0 else 0.0
        results_with_acc.append((idx, correct, total, accuracy))

    # Sort by accuracy descending
    results_with_acc.sort(key=lambda x: x[3], reverse=True)

    # Print all results
    for idx, correct, total, accuracy in results_with_acc:
        if total > 0:
            status = "✓ Best" if accuracy == results_with_acc[0][3] else ""
            print(f"{idx:<8} {correct:<10} {total:<10} {accuracy:>6.2f}%    {status}")

    print()
    print("=" * 100)

    # Find best index
    best_idx, best_correct, best_total, best_accuracy = results_with_acc[0]
    print(f"BEST: Index {best_idx} with {best_accuracy:.2f}% accuracy ({best_correct}/{best_total})")
    print("=" * 100)

    # Save detailed results
    output_file = base_dir / "baseline_result_all_indices.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                f'index_{idx}': {
                    'correct': index_results[idx]['correct'],
                    'total': index_results[idx]['total'],
                    'accuracy': (index_results[idx]['correct'] / index_results[idx]['total']
                                if index_results[idx]['total'] > 0 else 0.0)
                }
                for idx in range(32)
            },
            'detailed_results': detailed_results_per_index
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    print()

    return index_results


if __name__ == "__main__":
    evaluate_all_indices()
