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
Counterfactual Rollout for Reranker Training.

Key Idea:
- For each trajectory, create multiple branches at tool call positions
- Each branch samples different reranker outputs (tool responses)
- Continue generation from each branch
- Group samples by branch point for GRPO training
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict
from pydantic import BaseModel, ConfigDict, Field

import numpy as np

from verl.experimental.agent_loop.tool_parser import FunctionCall


@dataclass
class TrajectoryState:
    """State of a trajectory at a specific point."""
    messages: List[Dict[str, Any]]
    """Full message history up to this point."""

    executed_tool_calls: List[FunctionCall] = field(default_factory=list)
    
    is_valid: bool = True
    
    reward_score: float = None
    """EM/F1 score for this trajectory."""
    
    reward_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Reward computation kwargs from original Parquet data."""


class RerankerAgentData(BaseModel):
    """Data for reranker agent in counterfactual rollout."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    uid: str = Field(..., description="Unique identifier for GRPO grouping")
    prefix_messages: List[Dict[str, Any]] = Field(default_factory=list)
    tool_call: FunctionCall
    
    # Reranker agent outputs
    prompt_ids: Optional[List[int]] = None
    response_ids: Optional[List[int]] = None
    response_mask: Optional[List[int]] = None
    response_logprobs: Optional[List[float]] = None
    
    # Continue with search-r1 agent
    raw_messages: Optional[List[Dict[str, Any]]] = None
    user_turns: int = 0
    assistant_turns: int = 0
    
    # Reward computation kwargs (from original Parquet data)
    reward_kwargs: Optional[Dict[str, Any]] = None
    
    # Tool execution metrics (for tracking reranker success/fallback)
    tool_metrics: Optional[Dict[str, Any]] = None
    
    reward_score: Optional[float] = None
    
    # tool reward based on whether answer is found in retrieved docs
    tool_reward: float = 0.0
    search_r1_reward: float = -1.0

    # data for training 
    format_penalty: float = -0.2 
    is_success: bool = False
    
    # corner case for reranker 
    reranker_crashed: bool = False

    def should_train(self) -> bool:
        """Check if this sample should be used for training.
        
        Returns False if there was an execution_error (infra issue).
        Returns True for success or format_validation_error cases.
        """
        if not self.tool_metrics:
            return True  # No metrics, assume valid
        
        fallback_reason = self.tool_metrics.get("reranker_fallback_reason")
        if fallback_reason == "execution_error":
            return False  # Infra error, don't train
        
        return True  # Success or format error, both are trainable


# Helper function for Trajectory 
def validate_mssg_tool_call_match(trajectory: TrajectoryState) -> bool:
    """Validate that the number of executed tool calls matches tool messages."""
    tool_call_count = sum(1 for msg in trajectory.messages if msg["role"] == "tool")
    executed_tool_call_count = len(trajectory.executed_tool_calls or [])
    return tool_call_count == executed_tool_call_count
