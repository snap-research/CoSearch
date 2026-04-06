# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import asyncio
import copy
import json
import logging
import os
from typing import Any

from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SearchR1RewardLoop(RewardLoopManagerBase):
    """Reward loop for reranker agent.
    
    Workflow:
    1. Given reranker output (reranked document IDs), replace original tool response
    2. Call search-R1 agent with updated messages to get final answer
    3. Compare with golden answer to compute reward
    4. Optionally do multiple rollouts and aggregate rewards
    """

    def __init__(
        self, 
        config: DictConfig, 
        tokenizer: AutoTokenizer,
        search_r1_agent_loop,  # Search-R1 agent loop instance for generating final answer
        n_rollouts: int = 1,
        reward_aggregation: str = "max"  # "mean", "max", "median"
    ):
        super().__init__(config, tokenizer)
        self.search_r1_agent_loop = search_r1_agent_loop
        self.n_rollouts = n_rollouts
        self.reward_aggregation = reward_aggregation

    async def run_single(self, data: DataProto):
        """Compute reward for reranker output.
        
        Args:
            data: DataProto containing:
                - original_messages: [user, assistant, tool, assistant, tool]
                - reranked_documents: Top-M reranked documents
                - golden_answer: Expected answer for comparison
                - sampling_params: For search-R1 generation
        
        Returns:
            reward: Float reward value
        """
        original_messages = data.batch["original_messages"]
        reranked_documents = data.batch["reranked_documents"]
        golden_answer = data.batch.get("golden_answer", None)
        sampling_params = data.batch.get("sampling_params", {})
        
        # Replace last tool response with reranked documents
        updated_messages = self._replace_tool_response(original_messages, reranked_documents)
        
        # Run search-R1 multiple times in parallel if n_rollouts > 1
        tasks = [
            self._generate_and_compute_reward(updated_messages, golden_answer, sampling_params, i)
            for i in range(self.n_rollouts)
        ]
        rewards = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_rewards = []
        for i, reward in enumerate(rewards):
            if isinstance(reward, Exception):
                logger.error(f"Error in rollout {i+1}: {reward}")
                valid_rewards.append(0.0)
            else:
                valid_rewards.append(reward)
                logger.info(f"Rollout {i+1}/{self.n_rollouts}: reward={reward:.4f}")
        
        rewards = valid_rewards
        
        # Aggregate rewards
        if self.reward_aggregation == "mean":
            final_reward = sum(rewards) / len(rewards) if rewards else 0.0
        elif self.reward_aggregation == "max":
            final_reward = max(rewards) if rewards else 0.0
        elif self.reward_aggregation == "median":
            sorted_rewards = sorted(rewards)
            n = len(sorted_rewards)
            final_reward = sorted_rewards[n // 2] if n > 0 else 0.0
        else:
            raise ValueError(f"Unknown reward_aggregation: {self.reward_aggregation}")
        
        return final_reward

    def _replace_tool_response(self, original_messages: list[dict], reranked_documents: list[dict]) -> list[dict]:
        """Replace the last tool response with reranked documents.
        
        Args:
            original_messages: [user, assistant, tool, assistant, tool]
            reranked_documents: Top-M reranked documents
        
        Returns:
            Updated messages: [user, assistant, tool, assistant, tool']
        """
        updated_messages = copy.deepcopy(original_messages)
        
        # Find last tool message
        for i in range(len(updated_messages) - 1, -1, -1):
            if updated_messages[i].get("role") == "tool":
                # Format reranked documents
                formatted_docs = self._format_documents(reranked_documents)
                updated_messages[i]["content"] = formatted_docs
                break
        
        return updated_messages

    async def _generate_and_compute_reward(
        self, 
        messages: list[dict], 
        golden_answer: str, 
        sampling_params: dict,
        rollout_id: int
    ) -> float:
        """Generate final answer and compute reward for a single rollout.
        
        Args:
            messages: Updated messages with reranked tool response
            golden_answer: Expected golden answer
            sampling_params: Sampling parameters for generation
            rollout_id: Rollout index for logging
        
        Returns:
            Reward value
        """
        # Call search-R1 to get final answer
        final_answer = await self._generate_final_answer(messages, sampling_params)
        
        # Compute reward by comparing with golden answer
        reward = self._compute_reward(final_answer, golden_answer)
        
        return reward

    def _format_documents(self, documents: list[dict]) -> str:
        """Format documents as tool response text.
        
        Args:
            documents: List of documents with id, content, etc.
        
        Returns:
            Formatted string
        """
        if not documents:
            return "No relevant documents found."
        
        lines = ["Search Results (Reranked):", ""]
        for i, doc in enumerate(documents, 1):
            doc_id = doc.get("id", f"doc_{i}")
            content = doc.get("content", "")[:500]  # Truncate
            score = doc.get("score", 0.0)
            lines.append(f"{i}. [ID: {doc_id}] (score: {score:.4f})")
            lines.append(f"   {content}")
            lines.append("")
        
        return "\n".join(lines)

    async def _generate_final_answer(self, messages: list[dict], sampling_params: dict) -> str:
        """Use search-R1 agent to generate final answer.
        
        Args:
            messages: Updated messages with reranked tool response
            sampling_params: Sampling parameters for generation
        
        Returns:
            Final answer text
        """
        # Prepare input for search-R1 agent loop
        kwargs = {
            "raw_prompt": messages,
            "multi_modal_data": {},
            "tools_kwargs": {}
        }
        
        # Run search-R1 agent loop
        output = await self.search_r1_agent_loop.run(sampling_params, **kwargs)
        
        # Extract final answer from output
        # Decode response_ids to get final answer text
        final_answer = self.tokenizer.decode(output.response_ids, skip_special_tokens=True)
        
        return final_answer

    def _compute_reward(self, final_answer: str, golden_answer: str) -> float:
        """Compute reward by comparing final answer with golden answer.
        
        Args:
            final_answer: Generated answer from search-R1
            golden_answer: Expected golden answer
        
        Returns:
            Reward value (e.g., exact match = 1.0, else 0.0, or use similarity metrics)
        """
        if golden_answer is None:
            logger.warning("No golden answer provided, returning 0.0 reward")
            return 0.0
        
        # TODO: Implement more sophisticated reward computation
        # Options:
        # 1. Exact match
        # 2. F1 score on tokens
        # 3. Semantic similarity (e.g., using embeddings)
        # 4. LLM-as-judge
        
        # Simple exact match for now (case-insensitive)
        final_answer_normalized = final_answer.strip().lower()
        golden_answer_normalized = golden_answer.strip().lower()
        
        if final_answer_normalized == golden_answer_normalized:
            return 1.0
        
        # Token-level F1 score
        final_tokens = set(final_answer_normalized.split())
        golden_tokens = set(golden_answer_normalized.split())
        
        if not golden_tokens:
            return 0.0
        
        intersection = final_tokens & golden_tokens
        if not intersection:
            return 0.0
        
        precision = len(intersection) / len(final_tokens) if final_tokens else 0.0
        recall = len(intersection) / len(golden_tokens) if golden_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
