"""
MINERVA task module using math_verify for evaluation.
"""

from .env import Env, extract_answer, extract_groundtruth, judge_correct


def get_train_test_dataset(split: str = "test"):
    """
    Get train/test dataset for MINERVA.
    Note: This is a placeholder - actual dataset loading should be implemented
    based on your MINERVA dataset location.
    """
    raise NotImplementedError(
        "MINERVA dataset loading should be implemented based on your dataset location. "
        "Update this function to load from your MINERVA dataset file."
    )
