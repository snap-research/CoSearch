# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
UID grouping functions for reranker rollout training.

These functions assign UIDs to outputs for GRPO training.
UIDs are used to group counterfactual rollouts together.
"""

import re
import logging
from collections import Counter
from typing import List, Any

logger = logging.getLogger(__name__)


def normalize_subquery(s: str) -> str:
    """Normalize sub-query for comparison.
    
    Args:
        s: Sub-query string.
        
    Returns:
        Normalized string (lowercase, no articles, no punctuation).
    """
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)  # Remove punctuation
    s = " ".join(s.split())  # Normalize whitespace
    return s


def rouge1_f1(candidate: str, reference: str) -> float:
    """Compute ROUGE-1 F1 score between two strings.
    
    Both inputs should already be normalized.
    
    Args:
        candidate: Candidate string (normalized).
        reference: Reference string (normalized).
        
    Returns:
        ROUGE-1 F1 score (0.0 to 1.0).
    """
    cand_tokens = candidate.split()
    ref_tokens = reference.split()

    if len(cand_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    cand_counts = Counter(cand_tokens)
    ref_counts = Counter(ref_tokens)

    # Unigram overlap
    overlap = sum(
        min(cand_counts[token], ref_counts[token])
        for token in cand_counts
        if token in ref_counts
    )

    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def group_by_muid_ans_in_doc(outputs: List[Any], **kwargs) -> None:
    """Assign UID based on main_uid + answer_in_docs.
    
    This is the default/baseline grouping strategy.
    
    UID format: {main_uid}_easy  (if answer_in_docs=True)
                {main_uid}_hard  (if answer_in_docs=False)
    
    Args:
        outputs: List of AgentLoopOutput objects.
        **kwargs: Additional arguments (unused).
    
    Modifies:
        Sets output.extra_fields["uid"] for each output.
    """
    for output in outputs:
        main_uid = output.extra_fields["main_uid"]
        answer_in_docs = output.extra_fields.get("answer_in_docs", False)
        
        # UID format: main_uid + "_easy" or "_hard"
        uid = f"{main_uid}_{'easy' if answer_in_docs else 'hard'}"
        output.extra_fields["uid"] = uid


def group_by_muid_ans_in_doc_subq_em(outputs: List[Any], **kwargs) -> None:
    """Assign UID based on main_uid + answer_in_docs + exact match of normalized sub_query.
    
    This adds finer-grained grouping: within each (main_uid, answer_in_docs) pair,
    further group by exact match of normalized sub-query.
    
    UID format: {main_uid}_easy_{normalized_subquery}
                {main_uid}_hard_{normalized_subquery}
    
    Args:
        outputs: List of AgentLoopOutput objects.
        **kwargs: Additional arguments (unused).
    
    Modifies:
        Sets output.extra_fields["uid"] for each output.
    """
    for output in outputs:
        main_uid = output.extra_fields["main_uid"]
        answer_in_docs = output.extra_fields.get("answer_in_docs", False)
        
        # Get sub-query from extra_fields
        sub_query = output.extra_fields.get("sub_query", "")
        if not sub_query:
            logger.error(f"Output missing sub_query for UID grouping.")
        
        # Normalize sub-query for exact match grouping
        normalized_subquery = normalize_subquery(sub_query)
        
        # Replace spaces with underscores for UID safety
        normalized_subquery = normalized_subquery.replace(" ", "_")
        
        # UID format: main_uid + answer_in_docs + normalized_subquery
        uid = f"{main_uid}_{'easy' if answer_in_docs else 'hard'}_{normalized_subquery}"
        output.extra_fields["uid"] = uid


def group_by_muid_ans_in_doc_subq_rougeL1(
    outputs: List[Any], 
    threshold: float = 0.8, 
    **kwargs
) -> None:
    """Assign UID based on main_uid + answer_in_docs + ROUGE-1 F1 clustering of sub_queries.
    
    This uses ROUGE-1 F1 similarity to cluster sub-queries within each (main_uid, answer_in_docs) pair.
    Sub-queries with similarity >= threshold are grouped together.
    
    Uses greedy clustering: each sub-query is compared to representatives of existing clusters.
    
    UID format: {main_uid}_easy_cluster_{cluster_id}
                {main_uid}_hard_cluster_{cluster_id}
    
    Args:
        outputs: List of AgentLoopOutput objects.
        threshold: ROUGE-1 F1 threshold for clustering (default: 0.8).
        **kwargs: Additional arguments (unused).
    
    Modifies:
        Sets output.extra_fields["uid"] for each output.
    """
    # First, group by (main_uid, answer_in_docs)
    from collections import defaultdict
    base_groups = defaultdict(list)
    
    for output in outputs:
        main_uid = output.extra_fields["main_uid"]
        answer_in_docs = output.extra_fields.get("answer_in_docs", False)
        base_key = f"{main_uid}_{'easy' if answer_in_docs else 'hard'}"
        base_groups[base_key].append(output)
    
    # Within each base group, cluster by ROUGE-1 F1 similarity
    for base_key, group_outputs in base_groups.items():
        # Extract and normalize sub-queries
        subqueries = []
        for output in group_outputs:
            sub_query = output.extra_fields.get("sub_query", "")
            if not sub_query:
                logger.error(f"Output missing sub_query for UID grouping.")
            normalized = normalize_subquery(sub_query)
            subqueries.append(normalized)
        
        # Greedy clustering using ROUGE-1 F1
        clusters = {}  # cluster_id -> representative normalized sub-query
        output_to_cluster = {}  # index -> cluster_id
        
        for i, norm_subquery in enumerate(subqueries):
            # Check if this sub-query can be merged into an existing cluster
            merged = False
            for cluster_id, representative in clusters.items():
                similarity = rouge1_f1(norm_subquery, representative)
                
                if similarity >= threshold:
                    # Merge into this cluster
                    output_to_cluster[i] = cluster_id
                    merged = True
                    break
            
            if not merged:
                # Create new cluster
                cluster_id = len(clusters)
                clusters[cluster_id] = norm_subquery  # Use this as representative
                output_to_cluster[i] = cluster_id
        
        # Assign UIDs based on cluster assignments
        for i, output in enumerate(group_outputs):
            cluster_id = output_to_cluster[i]
            uid = f"{base_key}_cluster_{cluster_id}"
            output.extra_fields["uid"] = uid
