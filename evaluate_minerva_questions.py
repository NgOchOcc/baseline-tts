"""
Evaluate 272 MINERVA questions using majority voting and prm_last_max strategies.
Uses standard extract_boxed_answer and verify_answer from minerva.py via minerva_utils.py
"""

import os
import json
import re
from collections import Counter
from typing import List, Optional, Tuple
from pathlib import Path

# Import standard utilities from minerva.py
from minerva_utils import extract_boxed_answer, verify_answer

# ── Helper functions ──────────────────────────────────────────────────────

def majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    """Return the most common non-None answer (ties → first)."""
    valid = [a for a in answers if a]
    if not valid:
        return None
    counts = Counter(valid)
    return counts.most_common(1)[0][0]


def get_prm_last_max_answer(outputs: List[dict]) -> Optional[str]:
    """
    Select the output with highest last value in reward_history.
    Return its extracted_answer.
    """
    if not outputs:
        return None

    best_output = None
    best_reward = float('-inf')

    for output in outputs:
        if 'reward_history' in output and output['reward_history']:
            last_reward = output['reward_history'][-1]
            if last_reward > best_reward:
                best_reward = last_reward
                best_output = output

    if best_output and 'extracted_answer' in best_output:
        return best_output['extracted_answer']

    return None


# ── Main evaluation ────────────────────────────────────────────────────────

def evaluate_minerva_questions():
    base_dir = Path("/Users/luungoc/Project/compute-optimal-tts/MINERVA_best_of_n")
    base_dir = base_dir / "Qwen2.5-7B-Instruct" / "Qwen2.5-Math-PRM-7B" / "seed_0_width_32_num_seq_32_num_q_0"

    # Get all question directories
    question_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("question_")],
                          key=lambda x: int(x.name.split("_")[1]))

    print(f"Found {len(question_dirs)} question directories")
    print("=" * 80)

    # Track results
    results = {
        'majority_vote': {'correct': 0, 'total': 0},
        'prm_last_max': {'correct': 0, 'total': 0},
    }

    detailed_results = []

    # Process each question
    for idx, question_dir in enumerate(question_dirs):
        record_path = question_dir / "record_0.jsonl"

        if not record_path.exists():
            print(f"[{idx}] {question_dir.name}: record_0.jsonl not found, skipping")
            continue

        try:
            with open(record_path, 'r', encoding='utf-8') as f:
                # JSONL file: one JSON object per line
                line = f.readline().strip()
                if not line:
                    print(f"[{idx}] {question_dir.name}: Empty record file, skipping")
                    continue
                record = json.loads(line)
        except Exception as e:
            print(f"[{idx}] {question_dir.name}: Error reading JSON - {e}")
            continue

        # Extract data
        question = record.get('question', '')
        ground_truth = record.get('groundtruth', '')
        outputs = record.get('output', [])

        if not outputs or not ground_truth:
            print(f"[{idx}] {question_dir.name}: Missing outputs or groundtruth, skipping")
            continue

        # ── Majority voting ────────────────────────────────────────────────────
        extracted_answers = [o.get('extracted_answer') for o in outputs]
        mv_answer = majority_vote(extracted_answers)

        if mv_answer is not None:
            mv_correct = verify_answer(f"\\boxed{{{mv_answer}}}", ground_truth)
        else:
            mv_correct = False

        # ── PRM Last Max ───────────────────────────────────────────────────────
        prm_answer = get_prm_last_max_answer(outputs)

        if prm_answer is not None:
            prm_correct = verify_answer(f"\\boxed{{{prm_answer}}}", ground_truth)
        else:
            prm_correct = False

        # Update results
        results['majority_vote']['total'] += 1
        results['prm_last_max']['total'] += 1

        if mv_correct:
            results['majority_vote']['correct'] += 1
        if prm_correct:
            results['prm_last_max']['correct'] += 1

        # Store detailed result
        detailed_results.append({
            'idx': idx,
            'question_dir': question_dir.name,
            'ground_truth': ground_truth,
            'mv_answer': mv_answer,
            'mv_correct': mv_correct,
            'prm_answer': prm_answer,
            'prm_correct': prm_correct,
        })

        # Print progress every 20 questions
        if (idx + 1) % 20 == 0 or (idx + 1) == len(question_dirs):
            mv_acc = results['majority_vote']['correct'] / results['majority_vote']['total'] * 100
            prm_acc = results['prm_last_max']['correct'] / results['prm_last_max']['total'] * 100
            print(f"[{idx+1}/{len(question_dirs)}]  "
                  f"MV acc={mv_acc:.2f}%  PRM acc={prm_acc:.2f}%")

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    total = results['majority_vote']['total']
    mv_correct = results['majority_vote']['correct']
    prm_correct = results['prm_last_max']['correct']

    print(f"Total questions:                    {total}")
    print(f"Majority Vote accuracy:             {mv_correct}/{total} = {mv_correct/total*100:.2f}%")
    print(f"PRM Last Max accuracy:              {prm_correct}/{total} = {prm_correct/total*100:.2f}%")
    print("=" * 80)

    # Save detailed results
    output_file = Path("/Users/luungoc/Project/compute-optimal-tts/evaluation_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': total,
                'majority_vote_correct': mv_correct,
                'majority_vote_accuracy': mv_correct/total*100,
                'prm_last_max_correct': prm_correct,
                'prm_last_max_accuracy': prm_correct/total*100,
            },
            'detailed_results': detailed_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_file}")

    return results


if __name__ == "__main__":
    evaluate_minerva_questions()
