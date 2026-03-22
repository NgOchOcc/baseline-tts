"""
Detailed comparison of voting strategies for MINERVA questions.
Shows which strategy wins on each question and why.
Uses standard utilities from minerva.py via minerva_utils.py
"""

import os
import json
from collections import Counter, defaultdict
from typing import List, Optional, Dict, Tuple
from pathlib import Path

# Import standard utilities
from minerva_utils import verify_answer, verify_answer_smart

# Import voting functions
from minerva_voting_evaluation import (
    VOTING_STRATEGIES,
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_MAX,
    PRM_LAST_VOTE,
    PRM_AVG_MAX,
    PRM_AVG_VOTE,
    AGG_FN_MAP,
)


def compare_strategies():
    base_dir = Path("/Users/luungoc/Project/compute-optimal-tts/MINERVA_best_of_n")
    base_dir = base_dir / "Qwen2.5-7B-Instruct" / "Qwen2.5-Math-PRM-7B" / "seed_0_width_32_num_seq_32_num_q_0"

    # Get all question directories
    question_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("question_")],
                          key=lambda x: int(x.name.split("_")[1]))

    print("Comparing voting strategies per question...")
    print("=" * 100)

    # Track win counts
    win_counts = {strategy: 0 for strategy in VOTING_STRATEGIES}

    # Track specific cases
    cases = {
        'all_agree': [],  # All strategies select same answer and all correct
        'all_disagree': [],  # All strategies select different answers
        'voting_better': [],  # Voting strategies win over max strategies
        'max_better': [],  # Max strategies win over voting
        'only_one_correct': [],  # Only one strategy correct
    }

    strategy_correctness = {strategy: {'correct': 0, 'total': 0} for strategy in VOTING_STRATEGIES}

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

        # Prepare data for voting
        x_list = [o.get('extracted_answer') for o in outputs]
        v_list = [o.get('reward_history', []) for o in outputs]

        # Apply each voting strategy
        strategy_answers = {}
        strategy_correctness_map = {}

        for strategy in VOTING_STRATEGIES:
            try:
                if strategy == MAJORITY_VOTE:
                    selected_answer = AGG_FN_MAP[strategy](x_list, v_list)
                else:
                    selected_answer = AGG_FN_MAP[strategy](x_list, v_list)

                strategy_answers[strategy] = selected_answer
                is_correct = verify_answer_smart(f"\\boxed{{{selected_answer}}}", ground_truth) if selected_answer else False
                strategy_correctness_map[strategy] = is_correct

                if is_correct:
                    strategy_correctness[strategy]['correct'] += 1
                    win_counts[strategy] += 1

                strategy_correctness[strategy]['total'] += 1

            except Exception:
                strategy_answers[strategy] = None
                strategy_correctness_map[strategy] = False
                strategy_correctness[strategy]['total'] += 1

        # Categorize question
        all_answers = set(a for a in strategy_answers.values() if a)
        correct_count = sum(1 for v in strategy_correctness_map.values() if v)

        if len(all_answers) == 1 and all(strategy_correctness_map.values()):
            cases['all_agree'].append({
                'idx': idx,
                'answer': list(all_answers)[0],
                'gt': ground_truth,
            })

        elif len(all_answers) == len(VOTING_STRATEGIES):
            cases['all_disagree'].append({
                'idx': idx,
                'answers': {s: strategy_answers[s] for s in VOTING_STRATEGIES},
                'correctness': strategy_correctness_map,
                'gt': ground_truth,
            })

        if correct_count == 1:
            correct_strategy = [s for s in VOTING_STRATEGIES if strategy_correctness_map[s]][0]
            cases['only_one_correct'].append({
                'idx': idx,
                'strategy': correct_strategy,
                'answer': strategy_answers[correct_strategy],
                'gt': ground_truth,
            })

    # Print summary
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 100 + "\n")

    for strategy in sorted(VOTING_STRATEGIES, key=lambda s: strategy_correctness[s]['correct'], reverse=True):
        total = strategy_correctness[strategy]['total']
        correct = strategy_correctness[strategy]['correct']
        acc = correct / total * 100 if total > 0 else 0
        print(f"{strategy:20} {correct:3d}/{total} = {acc:5.2f}%")

    print("\n" + "-" * 100)
    print("SPECIFIC CASES ANALYSIS")
    print("-" * 100 + "\n")

    print(f"1. All strategies agree (same answer, all correct): {len(cases['all_agree'])} questions")
    print(f"   - These are the easiest questions where any strategy works\n")

    print(f"2. All strategies disagree (different answers): {len(cases['all_disagree'])} questions")
    if cases['all_disagree']:
        print(f"   - Example:")
        for case in cases['all_disagree'][:1]:
            print(f"     GT: {case['gt']}")
            for s, ans in case['answers'].items():
                correct_mark = "✓" if case['correctness'][s] else "✗"
                print(f"     {s:20} {ans:30} {correct_mark}")

    print(f"\n3. Only one strategy correct: {len(cases['only_one_correct'])} questions")
    if cases['only_one_correct']:
        print(f"   - These show when voting helps vs. max selection")
        strategy_wise = defaultdict(int)
        for case in cases['only_one_correct']:
            strategy_wise[case['strategy']] += 1
        print(f"   - Breakdown by winning strategy:")
        for strategy, count in sorted(strategy_wise.items(), key=lambda x: x[1], reverse=True):
            print(f"     {strategy:20} {count:3d} questions")

    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100 + "\n")

    # Voting vs Max analysis
    voting_strategies = [MAJORITY_VOTE, PRM_MIN_VOTE, PRM_LAST_VOTE, PRM_AVG_VOTE]
    max_strategies = [PRM_MIN_MAX, PRM_LAST_MAX, PRM_AVG_MAX]

    voting_correct = sum(strategy_correctness[s]['correct'] for s in voting_strategies)
    max_correct = sum(strategy_correctness[s]['correct'] for s in max_strategies)

    print(f"Voting strategies total correct:     {voting_correct}/1088 = {voting_correct/1088*100:.2f}%")
    print(f"Max-selection strategies total:      {max_correct}/816  = {max_correct/816*100:.2f}%")
    print(f"Voting strategies WIN by {(voting_correct/1088 - max_correct/816)*100:.2f} percentage points")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    compare_strategies()
