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
Score assignment functions for reranker rollout training.

These functions compute the final score from tool_score and agent_score.
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def max_tool_agent_score(
    tool_score: float,
    agent_score: float,
    agent_threshold: float = 0.8,
    **kwargs
) -> float:
    """Compute final score as max of tool_score and binarized agent_score.
    
    This is the default/baseline scoring strategy.
    
    Logic:
    - If tool_score < 0 (format penalty), return tool_score
    - Otherwise, return max(tool_score, binarized_agent_score)
    - binarized_agent_score = 1.0 if agent_score >= agent_threshold else 0.0
    
    Args:
        tool_score: Score from tool (reranker).
        agent_score: Score from agent continuation.
        agent_threshold: Threshold for binarizing agent_score (default: 0.8).
        **kwargs: Additional arguments (unused).
    
    Returns:
        Final aggregated score.
    """
    # Format penalty (negative tool score indicates format penalty)
    if tool_score < 0:
        return tool_score
    
    # Binarize agent score
    binarized_agent_score = 1.0 if agent_score >= agent_threshold else 0.0
    
    # Return max
    return max(tool_score, binarized_agent_score)


def sum_tool_agent_score_with_cond_threshold(
    tool_score: float,
    agent_score: float,
    answer_in_docs: bool,
    agent_threshold: float = 0.8,
    cond_threshold: float = 0.5,
    **kwargs
) -> float:
    """Compute final score as conditional sum of tool_score and binarized agent_score.
    
    This uses conditional logic based on answer_in_docs and tool_score threshold.
    
    Logic:
    1. Binarize agent_score first
    2. If answer_in_docs is True:
       - If tool_score < cond_threshold, return tool_score only
       - Otherwise, return tool_score + binarized_agent_score
    3. If answer_in_docs is False:
       - If tool_score < 0 (format wrong), return tool_score only
       - Otherwise, return tool_score + binarized_agent_score
    
    Args:
        tool_score: Score from tool (reranker).
        agent_score: Score from agent continuation.
        answer_in_docs: Whether answer is in retrieved documents.
        agent_threshold: Threshold for binarizing agent_score (default: 0.8).
        cond_threshold: Conditional threshold for answer_in_docs=True case (default: 0.5).
        **kwargs: Additional arguments (unused).
    
    Returns:
        Final aggregated score.
    """
    # Binarize agent score
    binarized_agent_score = 1.0 if agent_score >= agent_threshold else 0.0
    
    if answer_in_docs:
        # Easy case: answer is in docs
        if tool_score < cond_threshold:
            return tool_score
        else:
            return tool_score + binarized_agent_score
    else:
        # Hard case: answer not in docs
        if tool_score < 0:  # Format wrong
            return tool_score
        else:
            return tool_score + binarized_agent_score

def tool_score_only_with_format_penalty(
    tool_score: float,
    **kwargs
) -> float:
    """Use tool_score only, which already incorporates format penalty.
    
    This is a simpler variant that relies entirely on the tool_score for reward assignment.
    
    Logic:
    - Return tool_score directly (assumes it already includes any format penalties).
    
    Args:
        tool_score: Score from tool (reranker), expected to include format penalty if applicable.
        **kwargs: Additional arguments (unused).
    """
    return tool_score

def combine_tool_llm_judge_score(tool_score: float, 
                                 llm_judge_score: float = 0.0, 
                                 only_use_llm_judge: bool = True,
                                 **kwargs) -> float:
    """Combine tool_score with llm_judge_score for per-step reward.
    
    This is Variant A: per-step scoring using LLM judge instead of agent_score.
    
    Logic:
    - If tool_score < 0 (format penalty), return tool_score
    - If only_use_llm_judge=True, return llm_judge_score directly
    - Otherwise, return average of tool_score and llm_judge_score
    
    Args:
        tool_score: Score from tool (reranker).
        llm_judge_score: Score from LLM-as-Judge (0-1 normalized).
        only_use_llm_judge: If True, use llm_judge_score only (ignoring tool_score).
        **kwargs: Additional arguments (agent_score, answer_in_docs, output, etc.).
    
    Returns:
        Final score.
    """
    if tool_score < 0:
        # Format penalty case, trust tool_score only
        return tool_score
    else:
        # Combine scores
        if only_use_llm_judge:
            return llm_judge_score
        else:
            # Simple average (can be tuned)
            return (tool_score + llm_judge_score) / 2.0



