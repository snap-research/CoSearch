"""
Custom Reward Functions for Reranker Training

This module provides reward compute_score functions that can be selected
via VERL's custom_reward_function configuration.

Each function follows the VERL signature:
    compute_score(data_source, solution_str, ground_truth, extra_info) -> float or dict

Usage in config:
    custom_reward_function:
        path: /path/to/reranker_reward_functions.py
        name: compute_mrr_penalty_score  # or compute_mrr_recall_penalty_score or compute_rank_eval_score
        reward_kwargs:
            format_penalty: -0.2
            recall_weight: 0.5  # only for mrr_recall_penalty
"""

# IMPORTANT: This file is loaded dynamically by VERL's get_custom_reward_fn()
# using importlib.util.spec_from_file_location, which means:
# - The module has no __package__ attribute
# - Relative imports (from .custom_rewards) will fail
# - Must use absolute imports from the verl package

from verl.utils.reward_score.custom_rewards import reranker_rewards


def compute_mrr_penalty_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    format_penalty: float = -0.2,
    **kwargs
):
    """
    Compute MRR-based reward with format penalty.
    
    Reward = MRR@M if valid format, else format_penalty
    
    Args:
        data_source: Dataset identifier (e.g., "msmarco_dev", "trec19")
        solution_str: Model-generated response string
        ground_truth: Relevance judgments dict {docid: relevance}
        extra_info: Additional info dict with keys:
            - model_type: str
            - docids: List[str]
            - qrels: dict (same as ground_truth)
            - N: int (total candidates)
            - M: int (top-M to select)
        format_penalty: Penalty for invalid format (default: -0.2)
        **kwargs: Additional keyword arguments (ignored)
    
    Returns:
        dict with keys:
            - score: float reward value
            - mrr: float MRR@M score
            - format_valid: bool whether format is valid
            - selected_docids: List[str] extracted docids
    """
    return reranker_rewards.compute_train_score_mrr_penalty(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        format_penalty=format_penalty,
    )


def compute_mrr_recall_penalty_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    format_penalty: float = -0.2,
    recall_weight: float = 0.5,
    **kwargs
):
    """
    Compute reward as weighted combination of MRR and Recall with format penalty.
    
    Reward = recall_weight * Recall@M + (1 - recall_weight) * MRR@M
    
    This is useful for TREC datasets where recall is important.
    
    Args:
        data_source: Dataset identifier
        solution_str: Model-generated response string
        ground_truth: Relevance judgments dict {docid: relevance}
        extra_info: Additional info dict (must include 'binary_qrels' for recall)
        format_penalty: Penalty for invalid format (default: -0.2)
        recall_weight: Weight for recall component (default: 0.5)
        **kwargs: Additional keyword arguments (ignored)
    
    Returns:
        dict with keys:
            - score: float combined reward
            - mrr: float MRR@M
            - recall: float Recall@M
            - format_valid: bool
            - selected_docids: List[str]
    """
    return reranker_rewards.compute_train_score_mrr_recall_penalty(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        format_penalty=format_penalty,
        recall_weight=recall_weight,
    )

def compute_train_score_average_hit_penalty(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    format_penalty: float = -0.2,
    **kwargs
):
    """
    Compute reward as average hit rate with format penalty.
    
    Reward = Average Hit Rate if valid format, else format_penalty
    
    Args:
        data_source: Dataset identifier
        solution_str: Model-generated response string
        ground_truth: Relevance judgments dict {docid: relevance}
        extra_info: Additional info dict
        format_penalty: Penalty for invalid format (default: -0.2)
        **kwargs: Additional keyword arguments (ignored)
    
    Returns:
        dict with keys:
            - score: float reward value
            - average_hit_rate: float Average Hit Rate
            - format_valid: bool whether format is valid
            - selected_docids: List[str] extracted docids
    """
    return reranker_rewards.compute_train_score_average_hit_penalty(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        format_penalty=format_penalty,
    )


def compute_rank_eval_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    format_penalty: float = -0.2,
    **kwargs
):
    """
    Compute comprehensive ranking evaluation metrics.
    
    Used for validation/evaluation. Computes multiple metrics:
    - MRR@M, NDCG@M, P@M (for MS MARCO)
    - MRR@M, NDCG@M, Recall@M (for TREC with binary_qrels)
    
    Args:
        data_source: Dataset identifier
        solution_str: Model-generated response string
        ground_truth: Relevance judgments dict {docid: relevance}
        extra_info: Additional info dict
        format_penalty: Penalty for invalid format (default: -0.2)
        **kwargs: Additional keyword arguments (ignored)
    
    Returns:
        dict with keys:
            - score: float (MRR@M as primary metric)
            - mrr: float
            - ndcg: float
            - precision: float (MS MARCO) or recall: float (TREC)
            - format_valid: bool
            - selected_docids: List[str]
    """
    return reranker_rewards.compute_rank_eval_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )


if __name__ == "__main__":
    print("Reranker Reward Functions Module")
    print("=" * 60)
    print("\nAvailable functions:")
    print("  1. compute_mrr_penalty_score")
    print("     - MRR@M with format penalty")
    print("     - Best for: MS MARCO training")
    print()
    print("  2. compute_mrr_recall_penalty_score")
    print("     - Weighted MRR + Recall with format penalty")
    print("     - Best for: TREC training with recall optimization")
    print()
    print("  3. compute_rank_eval_score")
    print("     - Comprehensive metrics (MRR, NDCG, P/Recall)")
    print("     - Best for: Validation/Evaluation")
    print()
    print("Usage in VERL config:")
    print("  custom_reward_function:")
    print("    path: /path/to/reranker_reward_functions.py")
    print("    name: compute_mrr_penalty_score")
    print("    reward_kwargs:")
    print("      format_penalty: -0.2")
