"""
Reranker Reward Functions for Training

This module provides different reward computation strategies for reranker training.
Each function follows the signature: compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)

Available reward functions:
1. compute_train_score_mrr_penalty: MRR@M with format penalty (default for MS MARCO)
2. compute_train_score_mrr_recall: MRR@M + weighted Recall@M with format penalty (for MS MARCO)

Note: TREC datasets are used for evaluation only, not training.
      Use evaluate.py for TREC evaluation (supports NDCG@M, Recall@M).

Usage in config:
    custom_reward_function:
        path: reranker_rewards.py
        name: compute_train_score_mrr_penalty
        reward_kwargs:
            format_penalty: -0.2
"""

import json
import pytrec_eval
from typing import Dict, List, Optional, Any

from .validates import MODEL_TYPE_TO_VALIDATION


def _validate_output(solution_str: str, model_type: str, N: int, M: int, id_to_pid: Dict) -> Dict:
    """
    Validate reranker output format.
    
    Returns:
        Dict with 'error_type' (None if valid) and 'reranked' (list of docids)
    """
    validation_fn = MODEL_TYPE_TO_VALIDATION[model_type]
    return validation_fn(output=solution_str, N=N, M=M, id_to_pid=id_to_pid)

def binarize_qrel_for_trec(qrels: Dict[str, int]) -> Dict[str, int]:
        """Convert multi-graded qrels to binary for TREC datasets."""
        binary_qrels = {}
        for docid, rel in qrels.items():
            if rel >= 2:
                binary_qrels[docid] = 1
            else:
                binary_qrels[docid] = 0
        return binary_qrels


def _compute_mrr_at_m(run: Dict[str, float], qrels: Dict[str, int], M: int, data_source="dev") -> float:
    """
    Compute MRR@M using pytrec_eval.
    
    Args:
        run: {docid: score} ranking
        qrels: {docid: relevance} judgments
        M: Cutoff depth
    
    Returns:
        MRR@M score (0.0 to 1.0)
    """
    def binarize_qrel_for_trec(qrels: Dict[str, int]) -> Dict[str, int]:
        """Convert multi-graded qrels to binary for TREC datasets."""
        binary_qrels = {}
        for docid, rel in qrels.items():
            if rel >= 2:
                binary_qrels[docid] = 1
            else:
                binary_qrels[docid] = 0
        return binary_qrels
    
    if not run or not qrels:
        return 0.0
    
    # Manually truncate to top-M
    sorted_docs = sorted(run.items(), key=lambda x: -x[1])
    top_m_docs = sorted_docs[:M]
    truncated_run = {docid: score for docid, score in top_m_docs}
    
    # pytrec_eval format
    run_dict = {"q1": truncated_run}
    qrels_dict = {"q1": qrels}
    
    # if qrels is multi-graded like 0,1,2,3, binaryu it to 0,1, (0,1) -> 0, (2,3) -> 1 when data_source is trec19 or trec20
    if data_source in ["trec19", "trec20"]:
        qrels_dict["q1"] = binarize_qrel_for_trec(qrels)
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {'recip_rank'})
    results = evaluator.evaluate(run_dict)
    
    return float(results["q1"]['recip_rank'])

def _compute_average_hit_at_ks(run: Dict[str, float], qrels: Dict[str, int], hit_cutoffs: List[int]) -> float:
    """
    Compute Average Hit@{k1, k2, ...}.
    
    Args:
        run: {docid: score} ranking
        qrels: {docid: relevance} judgments
        hit_cutoffs: List of cutoff depths
    Returns:
        Average Hit@k score (0.0 to 1.0)
    """

    if not run or not qrels:
        return 0.0

    hit_scores = []
    for k in hit_cutoffs:
        hit_k = _compute_hit_at_m(run, qrels, k)
        hit_scores.append(hit_k)
    
    average_hit = sum(hit_scores) / len(hit_scores)
    return average_hit


def _compute_recall_at_m(run: Dict[str, float], qrels: Dict[str, int], M: int) -> float:
    """
    Compute Recall@M using pytrec_eval.
    
    Args:
        run: {docid: score} ranking
        qrels: {docid: relevance} judgments
        M: Cutoff depth
    
    Returns:
        Recall@M score (0.0 to 1.0)
    """
    if not run or not qrels:
        return 0.0
    
    # Manually truncate to top-M
    sorted_docs = sorted(run.items(), key=lambda x: -x[1])
    top_m_docs = sorted_docs[:M]
    truncated_run = {docid: score for docid, score in top_m_docs}
    
    # pytrec_eval format
    run_dict = {"q1": truncated_run}
    qrels_dict = {"q1": qrels}
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {f'recall.{M}'})
    results = evaluator.evaluate(run_dict)
    
    return float(results["q1"][f'recall_{M}'])


def _compute_ndcg_at_m(run: Dict[str, float], qrels: Dict[str, int], M: int) -> float:
    """
    Compute NDCG@M using pytrec_eval.
    
    Args:
        run: {docid: score} ranking
        qrels: {docid: relevance} judgments (can be graded relevance for TREC)
        M: Cutoff depth
    
    Returns:
        NDCG@M score (0.0 to 1.0)
    """
    if not run or not qrels:
        return 0.0
    
    # Manually truncate to top-M
    sorted_docs = sorted(run.items(), key=lambda x: -x[1])
    top_m_docs = sorted_docs[:M]
    truncated_run = {docid: score for docid, score in top_m_docs}
    
    # pytrec_eval format
    run_dict = {"q1": truncated_run}
    qrels_dict = {"q1": qrels}
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {f'ndcg_cut.{M}'})
    results = evaluator.evaluate(run_dict)
    
    return float(results["q1"][f'ndcg_cut_{M}'])

def _compute_hit_at_m(run: Dict[str, float], qrels: Dict[str, int], M: int) -> float:
    """
    Compute Hit@M.
    
    Args:
        run: {docid: score} ranking
        qrels: {docid: relevance} judgments
        M: Cutoff depth
    Returns:
        Hit@M score (0.0 to 1.0)
    """
    if not run or not qrels:
        return 0.0
    
    # Manually truncate to top-M
    sorted_docs = sorted(run.items(), key=lambda x: -x[1])
    top_m_docs = sorted_docs[:M]
    
    for docid, _ in top_m_docs:
        if qrels.get(docid, 0) > 0:
            return 1.0
    return 0.0


# ============================================================================
# REWARD FUNCTION 1: MRR@M with Format Penalty (Default)
# ============================================================================

def compute_train_score_mrr_penalty(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict] = None,
    format_penalty: float = -0.2,
    **kwargs
) -> float:
    """
    Compute reward score using MRR@M with format penalty.
    
    REWARD FORMULA:
        reward = format_penalty (if invalid format) OR MRR@M (if valid format)
    
    Args:
        data_source: Dataset name (e.g., "msmarco_dev")
        solution_str: Model output containing reranked passage indices
        ground_truth: Ground truth relevance information (qrels dict)
        extra_info: Additional information dict containing:
            - model_type: Model type string (REQUIRED)
            - docids: List of document IDs
            - qrels: Relevance judgments {docid: relevance}
            - N: Total number of candidates
            - M: Number to select and rerank
        format_penalty: Penalty for format violations (default: -0.2)
        **kwargs: Additional keyword arguments
    
    Returns:
        Reward score (format_penalty to 1.0)
    """
    if extra_info is None:
        raise ValueError("extra_info must be provided")
    
    # Extract required parameters
    model_type = extra_info['model_type']
    docids = extra_info['docids']
    N = extra_info['N']
    M = extra_info['M']
    qrels = ground_truth

    if not qrels:
        raise ValueError("No qrels available in extra_info or ground_truth")
    
    # Validate N matches docids
    if N != len(docids):
        raise ValueError(f"N ({N}) does not match number of provided docids ({len(docids)})")
    
    # Build id_to_pid mapping (1-indexed)
    id_to_pid = {i + 1: str(docid) for i, docid in enumerate(docids)}
    
    # Validate output format
    validation_result = _validate_output(solution_str, model_type, N, M, id_to_pid)
    
    # Check if format is correct
    if validation_result['error_type'] is not None:
       return {"score": format_penalty}  # Format is wrong
    
    # Extract reranked docids
    reranked_docids = validation_result['reranked']
    if not reranked_docids:
        raise RuntimeError("No valid ranking extracted despite validation passing")
    
    # Build run dict: {docid: score}
    run = {}
    for rank, docid in enumerate(reranked_docids):
        run[str(docid)] = len(reranked_docids) - rank
    
    # Ensure qrels are strings
    qrels = {str(k): v for k, v in qrels.items()}
    
    # Compute MRR@M
    ranking_score = _compute_mrr_at_m(run, qrels, M)
    
    return {"score": float(ranking_score)}


# ============================================================================
# REWARD FUNCTION 2: MRR@M + Weighted Recall@M with Format Penalty
# ============================================================================

def compute_train_score_mrr_recall_penalty(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict] = None,
    format_penalty: float = -0.2,
    recall_weight: float = 0.5,
    **kwargs
) -> float:
    """
    Compute reward score using weighted combination of MRR@M and Recall@M.
    
    REWARD FORMULA:
        reward = format_penalty (if invalid format) 
                 OR (1 - recall_weight) * MRR@M + recall_weight * Recall@M (if valid format)
    
    This encourages both ranking quality (MRR) and coverage (Recall).
    
    Args:
        data_source: Dataset name
        solution_str: Model output
        ground_truth: Ground truth qrels
        extra_info: Additional info dict
        format_penalty: Penalty for format violations (default: -0.2)
        recall_weight: Weight for recall component (default: 0.3)
        **kwargs: Additional keyword arguments
    
    Returns:
        Reward score (format_penalty to 1.0)
    """
    if extra_info is None:
        raise ValueError("extra_info must be provided")
    
    # Extract required parameters
    model_type = extra_info['model_type']
    docids = extra_info['docids']
    N = extra_info['N']
    M = extra_info['M']
    qrels = extra_info.get('qrels') or ground_truth
    
    if not qrels:
        raise ValueError("No qrels available")
    
    if N != len(docids):
        raise ValueError(f"N ({N}) does not match docids ({len(docids)})")
    
    # Build id_to_pid mapping
    id_to_pid = {i + 1: str(docid) for i, docid in enumerate(docids)}
    
    # Validate format
    validation_result = _validate_output(solution_str, model_type, N, M, id_to_pid)
    
    if validation_result['error_type'] is not None:
        return {"score": format_penalty}
    
    # Extract reranked docids
    reranked_docids = validation_result['reranked']
    if not reranked_docids:
        raise RuntimeError("No valid ranking extracted")
    
    # Build run dict
    run = {}
    for rank, docid in enumerate(reranked_docids):
        run[str(docid)] = len(reranked_docids) - rank
    
    # Ensure qrels are strings
    qrels = {str(k): v for k, v in qrels.items()}
    
    # Compute MRR@M and Recall@M
    mrr_score = _compute_mrr_at_m(run, qrels, M)
    recall_score = _compute_recall_at_m(run, qrels, M)
    
    # Weighted combination
    ranking_score = mrr_score + recall_weight * recall_score
    
    return {"score": float(ranking_score)}


# ============================================================================
# REWAD FUNCTION 3: Hit@{k1, k2, k3} with Format Penalty 
# ============================================================================
def compute_train_score_average_hit_penalty(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict] = None,
    hit_cutoffs: List[int] = [1,3,5],
    format_penalty: float = -0.2,
    **kwargs
) -> float:
    """
    Compute reward score using MRR@M with format penalty.
    
    REWARD FORMULA:
        reward = format_penalty (if invalid format) OR MRR@M (if valid format)
    
    Args:
        data_source: Dataset name (e.g., "msmarco_dev")
        solution_str: Model output containing reranked passage indices
        ground_truth: Ground truth relevance information (qrels dict)
        extra_info: Additional information dict containing:
            - model_type: Model type string (REQUIRED)
            - docids: List of document IDs
            - qrels: Relevance judgments {docid: relevance}
            - N: Total number of candidates
            - M: Number to select and rerank
        format_penalty: Penalty for format violations (default: -0.2)
        hit_cutoffs: List of cutoff values for Hit@k (default: [1,3,5])
        **kwargs: Additional keyword arguments
    
    Returns:
        Reward score (format_penalty to 1.0)
    """
    if extra_info is None:
        raise ValueError("extra_info must be provided")
    
    # Extract required parameters
    model_type = extra_info['model_type']
    docids = extra_info['docids']
    N = extra_info['N']
    M = extra_info['M']
    qrels = ground_truth

    assert M in [3, 5, 10], "M must be one of [3,5,10] for Hit@k reward"
    if M not in hit_cutoffs:
        hit_cutoffs.append(M)
        hit_cutoffs = sorted(hit_cutoffs)

    if not qrels:
        raise ValueError("No qrels available in extra_info or ground_truth")
    
    # Validate N matches docids
    if N != len(docids):
        raise ValueError(f"N ({N}) does not match number of provided docids ({len(docids)})")
    
    # Build id_to_pid mapping (1-indexed)
    id_to_pid = {i + 1: str(docid) for i, docid in enumerate(docids)}
    
    # Validate output format
    validation_result = _validate_output(solution_str, model_type, N, M, id_to_pid)
    
    # Check if format is correct
    if validation_result['error_type'] is not None:
        return {"score": format_penalty}  # Format is wrong
    
    # Extract reranked docids
    reranked_docids = validation_result['reranked']
    if not reranked_docids:
        raise RuntimeError("No valid ranking extracted despite validation passing")
    
    # Build run dict: {docid: score}
    run = {}
    for rank, docid in enumerate(reranked_docids):
        run[str(docid)] = len(reranked_docids) - rank
    
    # Ensure qrels are strings
    qrels = {str(k): v for k, v in qrels.items()}
    
    # Compute MRR@M
    ranking_score = _compute_average_hit_at_ks(run, qrels, hit_cutoffs)
    
    return {"score": float(ranking_score)}

def compute_rank_eval_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict] = None,
    **kwargs
):
    """
    Compute evaluation metrics based on dataset type.
    
    For TREC datasets (trec19, trec20):
        - Returns NDCG@M and Recall@M
    
    For MS MARCO dev:
        - Returns MRR@M and Recall@M

    For wiki datasets:
        - Returns Hit@{1,3,5} and Hit@M
    
    Args:
        data_source: Dataset name ("trec19", "trec20", "msmarco_dev", etc.)
        solution_str: Model output containing reranked passage indices
        ground_truth: Ground truth relevance information (qrels dict)
        extra_info: Additional information dict containing:
            - model_type: Model type string (REQUIRED)
            - docids: List of document IDs
            - qrels: Relevance judgments {docid: relevance}
            - N: Total number of candidates
            - M: Number to select and rerank
        **kwargs: Additional keyword arguments
    
    Returns:
        Dict with metric scores. Format violations return None for all metrics.
        - For TREC: {"ndcg": float, "recall": float, "valid": bool}
        - For MS MARCO: {"mrr": float, "recall": float, "valid": bool}
    """
    if extra_info is None:
        raise ValueError("extra_info must be provided")
    
    # Extract required parameters
    model_type = extra_info['model_type']
    docids = extra_info['docids']
    N = extra_info['N']
    M = extra_info['M']
    qrels = extra_info.get('qrels') or ground_truth
    
    if not qrels:
        raise ValueError("No qrels available in extra_info or ground_truth")
    
    # Validate N matches docids
    if N != len(docids):
        raise ValueError(f"N ({N}) does not match number of provided docids ({len(docids)})")
    
    # Build id_to_pid mapping (1-indexed)
    id_to_pid = {i + 1: str(docid) for i, docid in enumerate(docids)}
    
    # Validate output format
    validation_result = _validate_output(solution_str, model_type, N, M, id_to_pid)
    
    # Check if format is correct
    if validation_result['error_type'] is not None:
        assert M in [3, 5, 10], "M must be one of [3,5,10] for Hit@k reward"
        hit_cutoffs = [1, 3, 5]
        if M not in hit_cutoffs:
            hit_cutoffs.append(M)
            hit_cutoffs = sorted(hit_cutoffs)
        hit_scores = {f"hit@{k}": 0. for k in hit_cutoffs}
        result =  {"score": 0., f"mrr@{M}": 0., f"ndcg@{M}": 0., f"recall@{M}": 0., "valid": False}
        result.update({k: 0. for k in hit_scores.keys()})
        return result
    
    # Extract reranked docids
    reranked_docids = validation_result['reranked']
    if not reranked_docids:
        raise RuntimeError("No valid ranking extracted despite validation passing")
    
    # Build run dict: {docid: score}
    run = {}
    for rank, docid in enumerate(reranked_docids):
        run[str(docid)] = len(reranked_docids) - rank
    
    # Ensure qrels are strings
    qrels = {str(k): v for k, v in qrels.items()}
    assert M in [3, 5, 10], "M must be one of [3,5,10] for Hit@k reward"
    hit_cutoffs = [1, 3, 5]
    if M not in hit_cutoffs:
        hit_cutoffs.append(M)
        hit_cutoffs = sorted(hit_cutoffs)

    # Compute metrics based on dataset
    if data_source in ["trec19", "trec20"]:
        # TREC: NDCG@M and Recall@M
        mrr_score = _compute_mrr_at_m(run, qrels, M, data_source=data_source)
        ndcg_score = _compute_ndcg_at_m(run, qrels, M)
        recall_score = _compute_recall_at_m(run, qrels, M)
        hit_scores = {f"hit@{k}": _compute_hit_at_m(run, binarize_qrel_for_trec(qrels), k) for k in hit_cutoffs}
        
        result = {"score": float(ndcg_score),
                f"mrr@{M}": float(mrr_score),
                f"ndcg@{M}": float(ndcg_score), 
                f"recall@{M}": float(recall_score), "valid": True}
        result.update({k: float(v) for k, v in hit_scores.items()})
        return result
    
    elif data_source in ["nq", "triviaqa", "hotpotqa"]:
        # Wiki datasets: Hit@{1,3,5} and Hit@M
        mrr_score = _compute_mrr_at_m(run, qrels, M)
        ndcg_score = _compute_ndcg_at_m(run, qrels, M)
        recall_score = _compute_recall_at_m(run, qrels, M)
        hit_scores = {f"hit@{k}": _compute_hit_at_m(run, qrels, k) for k in hit_cutoffs}

        result = {"score": float(hit_scores[f"hit@{M}"])}
        result.update({k: float(v) for k, v in hit_scores.items()})
        result["valid"] = True
        result.update({f"mrr@{M}": float(mrr_score), 
                       f"ndcg@{M}": float(ndcg_score),
                       f"recall@{M}": float(recall_score)})
        return result
    else:
        # MS MARCO: MRR@M and Recall@M
        mrr_score = _compute_mrr_at_m(run, qrels, M)
        ndcg_score = _compute_ndcg_at_m(run, qrels, M)
        recall_score = _compute_recall_at_m(run, qrels, M)
        hit_scores = {f"hit@{k}": _compute_hit_at_m(run, qrels, k) for k in hit_cutoffs}
        result = {"score": float(mrr_score),
                f"mrr@{M}": float(mrr_score), 
                f"ndcg@{M}": float(ndcg_score),
                f"recall@{M}": float(recall_score), "valid": True}
        result.update({k: float(v) for k, v in hit_scores.items()})
        return result

# ============================================================================
# Default export (for backward compatibility)
# ============================================================================

# By default, use MRR@M with penalty
compute_train_score = compute_train_score_mrr_penalty


if __name__ == "__main__":
    # Test the reward functions
    print("=" * 80)
    print("Testing Reranker Reward Functions")
    print("=" * 80)
    
    # Mock test case
    test_output = """<reason>
Looking at the query and passages, passage 3 is most relevant.
</reason>
<select>[1], [2], [3]</select>
<rerank>[3] > [1] > [2]</rerank>"""
    
    test_docids = ["doc1", "doc2", "doc3", "doc4"]
    test_qrels = {"doc3": 1, "doc1": 0, "doc2": 0, "doc4": 0}
    
    extra_info = {
        "model_type": "qwen2507_distilled_instruct_thinking_rerank",
        "docids": test_docids,
        "qrels": test_qrels,
        "N": 4,
        "M": 3
    }
    
    print("\n[Test 1] MRR@M with penalty")
    print("-" * 80)
    reward1 = compute_train_score_mrr_penalty(
        data_source="msmarco_dev",
        solution_str=test_output,
        ground_truth=test_qrels,
        extra_info=extra_info,
        format_penalty=-0.2
    )
    print(f"Reward: {reward1:.4f}")
    print(f"Expected: 1.0 (relevant doc ranked first)")
    
    print("\n[Test 2] MRR@M + Recall@M (weight=0.3)")
    print("-" * 80)
    reward2 = compute_train_score_mrr_recall(
        data_source="msmarco_dev",
        solution_str=test_output,
        ground_truth=test_qrels,
        extra_info=extra_info,
        format_penalty=-0.2,
        recall_weight=0.3
    )
    print(f"Reward: {reward2:.4f}")
    print(f"Expected: 0.7 * 1.0 + 0.3 * 1.0 = 1.0")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
