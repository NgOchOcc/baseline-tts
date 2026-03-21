"""
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

import copy
import importlib
import jsonlines
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Union
import threading

import numpy as np
import os
import ray

from envs import get_default_query_str_builder, get_env_datasets
from envs.base_env import INVALID_ANS
from reason.inference.lm_call import LanguageModelCallingFunction, LMCallingConfig, ConcatedLMGenResult
from reason.inference.rm_call import RewardModelCallingFunction
from reason.reranking.vote_utils import (
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_MAX,
    PRM_LAST_VOTE,
    PRM_AVG_MAX,
    PRM_AVG_VOTE,
    AGG_FN_MAP,
)
from utils import get_step_cnt, to_raw_string, load_jsonl


class Task:
    def __init__(self, task_name: str, is_few_shot: bool = False, model_names=[]):
        # Map task aliases to actual task names
        if task_name in ("AMC23", "AIME24"):
            task_name = "MATH"

        self.task_name = task_name

        # Load task module dynamically
        try:
            task_module = importlib.import_module(f"envs.{task_name}")
        except ImportError:
            raise NotImplementedError(f"Task module 'envs.{task_name}' not found. Supported tasks: MATH, MINERVA")

        # Verify required functions exist
        required_functions = ["extract_answer", "extract_groundtruth", "judge_correct"]
        for func_name in required_functions:
            if not hasattr(task_module, func_name):
                raise NotImplementedError(
                    f"Task {task_name} module missing required function: {func_name}"
                )

        self.extract_answer = task_module.extract_answer
        self.extract_groundtruth = task_module.extract_groundtruth
        self.judge_correct = task_module.judge_correct

        self._is_few_shot = is_few_shot
        self.model_names = model_names
        self.env_fn = task_module.Env

    def prompt_fn(self, problem_input: str):
        # For MINERVA, use direct problem formatting without prompt builder
        if self.task_name == "MINERVA":
            return problem_input
        return get_default_query_str_builder(self.task_name)(problem_input, is_few_shot=self._is_few_shot, model_names=self.model_names)

    def test_ds(self, task_name):
        # For MINERVA, dataset loading should be handled by the calling code
        if task_name == "MINERVA":
            raise NotImplementedError("MINERVA dataset loading should be handled by the calling code")
        return get_env_datasets(task_name)[1]


CHOSEN_AGGR_METHODS = [
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_MAX,
    PRM_LAST_VOTE,
    PRM_AVG_MAX,
    PRM_AVG_VOTE,
]


def judge_ans(
    problem_str: str,
    extracted_groundtruth: str,
    extracted_answers: List[str],
    v_list: List[float],
    aggration_mode: str,
    judge_correct_fn,
    normalize=False,
):
    valid_ans_list, valid_v_list = [], []
    for i, ans in enumerate(extracted_answers):
        if ans != INVALID_ANS:
            valid_ans_list.append(ans)
            valid_v_list.append(v_list[i])
    if len(valid_ans_list) == 0:
        return 0

    if "orm" in aggration_mode and normalize:
        # score_normalization: this is only necessary for [-1, 1] values
        valid_v_list = np.array(valid_v_list)
        valid_v_list -= valid_v_list.min()
        valid_v_list /= valid_v_list.max() + 1e-3
        valid_v_list = valid_v_list.tolist()
    aggregated_ans = AGG_FN_MAP[aggration_mode](valid_ans_list, valid_v_list)

    return 1 if judge_correct_fn(problem_str, extracted_groundtruth, aggregated_ans) else 0


@dataclass
class SolutionOutput:
    solutions: List[str]
    completion_tokens: List[int]


@dataclass
class TreeSearchSolutionOutput(SolutionOutput):
    tree_completion_tokens: List[int]
    reward_history: List[float]
    token_history: List[int]
    prob_history: List[float]
    model_history: List[str]


class MathEvaluator:

    def __init__(
        self, task: Union[str, Task], lm_calls: List[LanguageModelCallingFunction], rm_call: RewardModelCallingFunction, direct_io=False, timeout_seconds: int = 300
    ):
        if isinstance(task, str):
            self._task = Task(task_name=task)
        else:
            assert isinstance(task, Task)
            self._task = task
        self.lm_calls = lm_calls
        self.rm_call = rm_call
        self.direct_io = direct_io
        self.timeout_seconds = timeout_seconds  # Default: 5 minutes (300 seconds)

    def evaluate_problem(self, problem_inst: Dict[str, str], solver_fn: Callable) -> List[str]:
        # Try to solve with timeout
        solution = None
        timeout_occurred = False

        def solve_with_timeout():
            nonlocal solution, timeout_occurred
            try:
                solution = solver_fn(problem_inst, self.lm_calls, self.rm_call)
            except Exception as e:
                print(f"Error during solving: {e}")
                traceback.print_exc()

        # Run solver in a thread with timeout
        solver_thread = threading.Thread(target=solve_with_timeout)
        solver_thread.daemon = True
        solver_thread.start()
        solver_thread.join(timeout=self.timeout_seconds)

        if solver_thread.is_alive():
            timeout_occurred = True
            print(f"TIMEOUT: Problem {problem_inst.get('question', '')[:50]}... took longer than {self.timeout_seconds} seconds. Marking as incorrect.")
            # Return empty solution (all incorrect)
            return self._get_timeout_result(problem_inst)

        if solution is None:
            print(f"ERROR: Failed to get solution for problem {problem_inst.get('question', '')[:50]}...")
            return self._get_timeout_result(problem_inst)

        reward_history = solution.reward_history
        token_history = solution.token_history
        prob_history = solution.prob_history
        if isinstance(solution, TreeSearchSolutionOutput):
            model_history = [[model.split('/')[-1] for model in traj] for traj in solution.model_history]
        else:
            model_history = [[]] * len(solution.solutions)
        result, output = self.analyze_output(problem_inst, solution.solutions, reward_history, token_history, prob_history, model_history)
        total_completion_token = 0
        for i, o in enumerate(output):
            o["completion_tokens"] = solution.completion_tokens[i]
            if isinstance(solution, TreeSearchSolutionOutput):
                o["tree_completion_tokens"] = solution.tree_completion_tokens[i]
            # We define the completion_tokens as the tokens consumed between two generated answers, therefore we need to take sum here.
            total_completion_token += solution.completion_tokens[i]
        result["total_completion_tokens"] = total_completion_token
        return problem_inst, result, output

    def _get_timeout_result(self, problem_inst: Dict[str, str]):
        """Return a result marking the problem as incorrect due to timeout."""
        # Create a minimal output indicating timeout
        result = {agg_method: 0 for agg_method in CHOSEN_AGGR_METHODS}
        result["total_completion_tokens"] = 0

        output = [{
            "path_idx": 0,
            "text": "[TIMEOUT] Solution took longer than {} seconds".format(self.timeout_seconds),
            "value": 0.0,
            "extracted_answer": "TIMEOUT",
            "reward_history": [0.0],
            "token_history": [0],
            "prob_history": [0.0],
            "model_history": [],
            "completion_tokens": 0,
            "tree_completion_tokens": 0,
        }]

        return problem_inst, result, output

    def analyze_output(
        self, problem_inst: Dict[str, str], gen_answers: List[str], reward_history, token_history, prob_history, model_history=None
    ):
        if 'extracted_groundtruth' in problem_inst:
            extracted_groundtruth = problem_inst['extracted_groundtruth']
        else:
            extracted_groundtruth = self._task.extract_groundtruth(problem_inst["answer"])

        if self.direct_io == 1:  # BoN
            input_list = [(problem_inst["question"], txt) for txt in gen_answers]
            for i in range(2):
                try:
                    value_list = self.rm_call(input_list)
                    break
                except Exception as e:
                    import traceback
                    print(f"Error in computing reward: {e}")
                    traceback.print_exc()
                    value_list = [[0.0]] * len(gen_answers)
            reward_history = value_list
        else:
            value_list = reward_history

        extracted_answers = [self._task.extract_answer(txt) for txt in gen_answers]
        output_list = [
            {
                "path_idx": i, "text": txt, "value": v, "extracted_answer": extracted_answer, "reward_history": reward,
                "token_history": token, "prob_history": prob, "model_history": model
            }
            for i, (txt, v, extracted_answer, reward, token, prob, model) in
            enumerate(zip(gen_answers, value_list, extracted_answers, reward_history, token_history, prob_history, model_history))
        ]
        res = {
            agg_method:
                judge_ans(
                    problem_inst["question"],
                    extracted_groundtruth,
                    extracted_answers,
                    value_list,
                    agg_method,
                    self._task.judge_correct,
                ) for agg_method in (CHOSEN_AGGR_METHODS if len(gen_answers) > 1 else [MAJORITY_VOTE])
        }
        return res, output_list


@ray.remote
class RemoteMathEvaluator(MathEvaluator):
    def __init__(
        self, task: str, lm_calls: List[LanguageModelCallingFunction], rm_call: RewardModelCallingFunction, direct_io=False, seed: int = None, timeout_seconds: int = 300
    ):
        super().__init__(task, lm_calls, rm_call, direct_io, timeout_seconds)
        # Set seed for this Ray actor process
        if seed is not None:
            from utils import setup_seed
            setup_seed(seed)
