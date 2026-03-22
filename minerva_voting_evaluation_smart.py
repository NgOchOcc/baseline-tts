"""
Comprehensive voting evaluation for MINERVA questions using SMART verification.
Applies all voting strategies from vote_utils.py with enhanced numeric comparison.
Uses verify_answer_smart for tolerance-based matching.
"""

import json
from collections import Counter, defaultdict
from typing import List, Optional
from pathlib import Path

# Import standard utilities
from minerva_utils import extract_boxed_answer, verify_answer_smart

# ── Voting Functions (from vote_utils.py) ──────────────────────────────────

MAJORITY_VOTE = "majority_vote"
PRM_MIN_MAX = "prm_min_max"
PRM_MIN_VOTE = "prm_min_vote"
PRM_LAST_MAX = "prm_last_max"
PRM_LAST_VOTE = "prm_last_vote"
PRM_AVG_MAX = "prm_avg_max"
PRM_AVG_VOTE = "prm_avg_vote"

VOTING_STRATEGIES = [
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_MAX,
    PRM_LAST_VOTE,
    PRM_AVG_MAX,
    PRM_AVG_VOTE,
]


def _agg_majority_vote(x_list: List[str], unused_v_list, return_reward=False):
    """Vote based on answer frequency."""
    counts = Counter(x_list)
    most_common = max(counts, key=counts.get)
    if return_reward:
        return most_common, [0.0]
    return most_common


def _agg_orm_vote(x_list: List[str], v_list: List[float], return_reward_idx=False):
    """Weighted vote by summing scores for each answer."""
    assert len(x_list) == len(v_list)
    x_dict = defaultdict(lambda: 0.0)
    for x, v in zip(x_list, v_list):
        x_dict[x] += v

    highest_x = max(x_dict, key=x_dict.get)
    if return_reward_idx:
        idx_list = [i for i, x in enumerate(x_list) if x == highest_x]
        corresponding_v_list = [v_list[idx] for idx in idx_list]
        idx = corresponding_v_list.index(max(corresponding_v_list))
        return highest_x, idx
    return highest_x


def _agg_prm_min_max(x_list: List[str], v_list: List[List[float]], return_reward=False):
    """Select output with highest minimum reward score."""
    new_v_list = [min(v) if v else -1.0 for v in v_list]
    idx = new_v_list.index(max(new_v_list))
    text_max = x_list[idx]
    if return_reward:
        return text_max, v_list[idx]
    return text_max


def _agg_prm_last_max(x_list: List[str], v_list: List[List[float]], return_reward=False):
    """Select output with highest last reward score."""
    new_v_list = [v[-1] if v else -1.0 for v in v_list]
    idx = new_v_list.index(max(new_v_list))
    text_max = x_list[idx]
    if return_reward:
        return text_max, v_list[idx]
    return text_max


def _agg_prm_min_vote(x_list: List[str], v_list: List[List[float]], return_reward=False):
    """Vote with weights = minimum reward of each output."""
    new_v_list = [min(v) if v else -1.0 for v in v_list]
    if return_reward:
        x, idx = _agg_orm_vote(x_list, new_v_list, return_reward_idx=True)
        return x, v_list[idx]
    return _agg_orm_vote(x_list, new_v_list)


def _agg_prm_last_vote(x_list: List[str], v_list: List[List[float]], return_reward=False):
    """Vote with weights = last reward of each output."""
    new_v_list = [v[-1] if v else -1.0 for v in v_list]
    if return_reward:
        x, idx = _agg_orm_vote(x_list, new_v_list, return_reward_idx=True)
        return x, v_list[idx]
    return _agg_orm_vote(x_list, new_v_list)


def _agg_prm_avg_max(x_list: List[str], v_list: List[List[float]], return_reward=False):
    """Select output with highest average reward score."""
    new_v_list = [(sum(v) / len(v)) if v else -1.0 for v in v_list]
    idx = new_v_list.index(max(new_v_list))
    text_max = x_list[idx]
    if return_reward:
        return text_max, v_list[idx]
    return text_max


def _agg_prm_avg_vote(x_list: List[str], v_list: List[List[float]], return_reward=False):
    """Vote with weights = average reward of each output."""
    new_v_list = [(sum(v) / len(v)) if v else -1.0 for v in v_list]
    if return_reward:
        x, idx = _agg_orm_vote(x_list, new_v_list, return_reward_idx=True)
        return x, v_list[idx]
    return _agg_orm_vote(x_list, new_v_list)


AGG_FN_MAP = {
    MAJORITY_VOTE: _agg_majority_vote,
    PRM_MIN_MAX: _agg_prm_min_max,
    PRM_MIN_VOTE: _agg_prm_min_vote,
    PRM_LAST_MAX: _agg_prm_last_max,
    PRM_LAST_VOTE: _agg_prm_last_vote,
    PRM_AVG_MAX: _agg_prm_avg_max,
    PRM_AVG_VOTE: _agg_prm_avg_vote,
}


def evaluate_all_strategies_smart():
    """Evaluate using verify_answer_smart (numeric tolerance, LaTeX handling)."""
    base_dir = Path("/Users/luungoc/Project/compute-optimal-tts/MINERVA_best_of_n")
    base_dir = base_dir / "Qwen2.5-7B-Instruct" / "Qwen2.5-Math-PRM-7B" / "seed_0_width_16_num_seq_16_num_q_0"

    # Get all question directories
    question_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("question_")],
        key=lambda x: int(x.name.split("_")[1])
    )

    print(f"Found {len(question_dirs)} question directories")
    print("=" * 80)
    print("USING: verify_answer_smart (EXACT numeric match + format normalization)")
    print("=" * 80)

    # Initialize tracking
    strategy_results = {strategy: {'correct': 0, 'total': 0} for strategy in VOTING_STRATEGIES}
    total_completion_tokens = 0

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

        # Count tokens
        total_completion_tokens += record.get('result', {}).get('total_completion_tokens', 0)

        # Apply each voting strategy
        for strategy in VOTING_STRATEGIES:
            try:
                if strategy == MAJORITY_VOTE:
                    selected_answer = AGG_FN_MAP[strategy](x_list, v_list)
                else:
                    selected_answer = AGG_FN_MAP[strategy](x_list, v_list)

                if selected_answer:
                    # Verify using smart comparison (exact numeric match + format normalization)
                    is_correct = verify_answer_smart(
                        f"\\boxed{{{selected_answer}}}",
                        ground_truth
                    )
                else:
                    is_correct = False

                strategy_results[strategy]['total'] += 1
                if is_correct:
                    strategy_results[strategy]['correct'] += 1

            except Exception:
                pass

        # Print progress
        if (idx + 1) % 50 == 0 or (idx + 1) == len(question_dirs):
            print(f"[{idx+1}/{len(question_dirs)}]")

    # Calculate accuracies
    results_dict = {}
    print()
    for strategy in VOTING_STRATEGIES:
        total = strategy_results[strategy]['total']
        correct = strategy_results[strategy]['correct']
        accuracy = correct / total if total > 0 else 0.0
        results_dict[strategy] = accuracy
        print(f"{strategy:20} {correct:3d}/{total} = {accuracy*100:6.2f}%")

    # Calculate average tokens
    total_questions = strategy_results[MAJORITY_VOTE]['total']
    avg_tokens = total_completion_tokens / total_questions if total_questions > 0 else 0

    results_dict['total_completion_tokens'] = avg_tokens

    print("\n" + "=" * 80)
    print(f"Average tokens per question: {avg_tokens:.2f}")
    print("=" * 80)

    # Save results to JSON
    output_file = base_dir / "avg_result_smart.json"
    with open(output_file, 'w') as f:
        json.dump([results_dict], f)

    print(f"\nResults saved to: {output_file}")
    return results_dict


if __name__ == "__main__":
    evaluate_all_strategies_smart()
