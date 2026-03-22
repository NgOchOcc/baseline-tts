"""
Baseline evaluation for MINERVA questions using SMART verification.
Simply takes output[0] (first output) as final answer and evaluates.
Uses verify_answer_smart for enhanced comparison (numeric tolerance, LaTeX).
"""

import json
from pathlib import Path
from minerva_utils import verify_answer_smart


def baseline_evaluation_smart():
    """Evaluate using output[0] as final answer with smart verification."""
    base_dir = Path("/Users/luungoc/Project/compute-optimal-tts/MINERVA_best_of_n")
    base_dir = base_dir / "Qwen2.5-7B-Instruct" / "Qwen2.5-Math-PRM-7B" / "seed_0_width_32_num_seq_32_num_q_0"

    # Get all question directories
    question_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("question_")],
        key=lambda x: int(x.name.split("_")[1])
    )

    print(f"Found {len(question_dirs)} question directories")
    print("=" * 80)
    print("BASELINE (SMART): Taking output[0] with verify_answer_smart")
    print("=" * 80)
    print()

    # Track results
    correct = 0
    total = 0
    detailed_results = []

    # Process each question
    for idx, question_dir in enumerate(question_dirs):
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

        # Baseline: Take output[0] only
        first_output = outputs[0]
        baseline_answer = first_output.get('extracted_answer')

        # Verify answer using smart verification
        if baseline_answer:
            is_correct = verify_answer_smart(
                f"\\boxed{{{baseline_answer}}}",
                ground_truth,
                tolerance=0.02  # 2% relative error
            )
        else:
            is_correct = False

        total += 1
        if is_correct:
            correct += 1

        # Store result
        detailed_results.append({
            'idx': idx,
            'question_dir': question_dir.name,
            'ground_truth': ground_truth,
            'baseline_answer': baseline_answer,
            'is_correct': is_correct,
        })

        # Print progress
        if (idx + 1) % 50 == 0 or (idx + 1) == len(question_dirs):
            acc = correct / total * 100 if total > 0 else 0
            print(f"[{idx+1}/{len(question_dirs)}]  Accuracy: {correct}/{total} = {acc:.2f}%")

    # Print summary
    print("\n" + "=" * 80)
    print("BASELINE (SMART) RESULTS")
    print("=" * 80)
    print()
    print(f"Total questions:  {total}")
    print(f"Correct:          {correct}")
    print(f"Accuracy:         {correct/total*100:.2f}%")
    print()
    print("=" * 80)

    # Save detailed results
    output_file = base_dir / "baseline_result_smart.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total': total,
                'correct': correct,
                'accuracy': correct / total if total > 0 else 0.0,
            },
            'detailed_results': detailed_results
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")
    print()

    # Compare with voting results if available
    avg_result_file = base_dir / "avg_result_smart.json"
    if avg_result_file.exists():
        print("=" * 80)
        print("COMPARISON WITH VOTING STRATEGIES (SMART)")
        print("=" * 80)
        print()

        with open(avg_result_file, 'r') as f:
            voting_results = json.load(f)[0]

        baseline_acc = correct / total * 100

        # Sort and display comparison
        print(f"{'Strategy':<20} {'Accuracy':<10} {'Diff vs Baseline':<20} {'Status'}")
        print("-" * 80)

        print(f"{'BASELINE (output[0])':<20} {baseline_acc:>6.2f}%   {'--':<20} ✅ Reference")
        print()

        # Voting strategies
        strategies = [
            ('majority_vote', 'Majority Vote'),
            ('prm_last_vote', 'PRM Last Vote'),
            ('prm_avg_vote', 'PRM Avg Vote'),
            ('prm_min_vote', 'PRM Min Vote'),
            ('prm_min_max', 'PRM Min Max'),
            ('prm_avg_max', 'PRM Avg Max'),
            ('prm_last_max', 'PRM Last Max'),
        ]

        for key, name in strategies:
            if key in voting_results:
                voting_acc = voting_results[key] * 100
                diff = voting_acc - baseline_acc
                status = "📈 Better" if diff > 0 else "📉 Worse" if diff < 0 else "🟰 Same"
                print(f"{name:<20} {voting_acc:>6.2f}%   {diff:+6.2f}%{'':<14} {status}")

        print()
        print("=" * 80)

    return correct, total


if __name__ == "__main__":
    baseline_evaluation_smart()
