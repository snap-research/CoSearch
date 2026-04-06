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
Extended DualAgentLoopManager with support for counterfactual rollout training.
"""

import asyncio
import logging
import os
from typing import Any, List, Dict, Optional
import uuid 
import copy
import random
import time 
import re 

import numpy as np
import ray
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import (
    get_trajectory_info,
    SearchR1DualAgentLoopWorkerBase,
    _DummyConfig,
)
from verl.experimental.agent_loop.counterfactual_rollout import (
    TrajectoryState,
)
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.rollout_trace import RolloutTraceConfig
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.experimental.agent_loop.counterfactual_rollout import (
    RerankerAgentData, 
    validate_mssg_tool_call_match)
from verl.utils.transferqueue_utils import tqbridge
from verl.utils.model import compute_position_id_with_mask
from verl.experimental.agent_loop.agent_loop import _InternalAgentLoopOutput
from verl.experimental.agent_loop.search_r1_agent_loop import SearchR1AgentLoop
from tensordict import TensorDict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

def extract_rerank_output(output_text: str) -> str:
    """Extract the content inside <rerank>...</rerank> tags."""
    match = re.search(r'<rerank>(.*?)</rerank>', output_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

@ray.remote
class SearchR1DualAgentLoopWorker(SearchR1DualAgentLoopWorkerBase):
    """Extended agent loop worker with counterfactual rollout support.
    
    This worker extends DualAgentLoopWorker to add counterfactual rollout capabilities
    for training the reranker agent with GRPO.
    """
    
    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
        reranker_server_handles: list[ray.actor.ActorHandle] = None,
    ):
        """Initialize extended agent loop worker.
        
        Args:
            config: YAML config.
            server_handles: Main agent server actor handles.
            reward_router_address: Reward router address.
            reranker_server_handles: Optional reranker server actor handles.
        """
        super().__init__(config, server_handles, reward_router_address, reranker_server_handles)
        
        # Semaphore to control max concurrent tasks (512 is safe based on testing)
        self.max_concurrent_tasks = 512
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        logger.info(f"Initialized worker with max_concurrent_tasks={self.max_concurrent_tasks}")

        self.reranker_main_train_n_ratio = config.trainer.get("reranker_main_train_n_ratio", 2)
        
        # Calculate reranker GPU count for FSDP alignment
        nnodes = config.trainer.nnodes
        n_gpus_per_node = config.trainer.n_gpus_per_node
        if nnodes == 1:
            # Single node: split GPUs on same node
            self.reranker_n_gpus = n_gpus_per_node // 2
        else:
            # Multi-node: reranker gets second half of nodes
            nodes_per_agent = nnodes // 2
            self.reranker_n_gpus = nodes_per_agent * n_gpus_per_node
        
        print(f"Reranker GPU count for FSDP alignment: {self.reranker_n_gpus}")
    
    async def _call_with_semaphore(self, func, data):
        """Execute async function with semaphore to limit concurrency."""
        async with self.semaphore:
            return await func(data)
    
    async def _process_reranker_pipeline(self, reranker_agent_data: RerankerAgentData) -> Optional[_InternalAgentLoopOutput]:
        """Process complete reranker pipeline for a single data point.
        
        This method chains two operations:
        1. Call reranker as agent (uses reranker GPUs)
        2. Continue with search-r1 agent (uses main agent GPUs + reranker GPUs)
        
        By pipelining these operations, we can utilize all 16 GPUs concurrently:
        - While task A is calling reranker (reranker GPUs busy), 
          task B can be calling search-r1 agent (main agent GPUs busy)
        
        Args:
            reranker_agent_data: Input data for reranker.
            
        Returns:
            Final output with reward score, or None if reranker crashed.
        """
        # Step 1: Call reranker tool as agent (uses reranker GPUs)
        updated_data = await self._call_reranker_tool_as_agent(reranker_agent_data)
        
        # Check if reranker crashed (infrastructure error) - skip continuation
        if updated_data.reranker_crashed:
            logger.warning(
                f"Reranker crashed for uid {reranker_agent_data.uid}. "
                f"Skipping search-r1 continuation to avoid wasting GPU resources."
            )
            return None
        
        # Step 2: Continue with search-r1 from reranker output (uses main agent GPUs)
        output = await self._continue_search_r1_from_reranker_agent_loop(updated_data)
        
        return output
        
    @tqbridge()
    async def generate_search_r1_initial_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.
        The code is copied from generate_sequences in AgentLoopWorkerBase that remove the postprocess step.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        kwargs_list = []  # Save kwargs for reward computation
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            kwargs_list.append(kwargs)
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)
        main_agent_batch = self._postprocess(outputs)

        # add answers and initial_question 
        assert len(outputs) == len(batch), "Output length mismatch!"


        trajectories = self._formalize_trajectory(outputs, kwargs_list)

        return main_agent_batch, trajectories

    def _formalize_trajectory(self, outputs: List[_InternalAgentLoopOutput], kwargs_list: List[Dict[str, Any]]) -> List[TrajectoryState]:
        """Convert agent loop outputs to TrajectoryState list.
        
        Args:
            outputs: List of _InternalAgentLoopOutput from agent loop.
            kwargs_list: List of kwargs containing reward computation information.
            
        Returns:
            List of TrajectoryState with reward_kwargs attached.
        """
        trajectories = []
        for output, kwargs in zip(outputs, kwargs_list):
            trajectory = TrajectoryState(
                messages=output.extra_fields["messages"],
                executed_tool_calls=output.extra_fields["executed_tool_calls"],
                reward_score=output.reward_score,
                is_valid=output.reward_score >= 0.0 # might change later. hard-code, as < 0 means format penalty only
            )
            # Attach reward_kwargs to trajectory for later use
            trajectory.reward_kwargs = {
                "data_source": kwargs.get("data_source"),
                "reward_model": kwargs.get("reward_model"),
                "ability": kwargs.get("ability"),
                "extra_info": kwargs.get("extra_info"),
                "index": kwargs.get("index"),
            }
            trajectories.append(trajectory)
        return trajectories

    def _construct_reranker_train_data(self, trajectory: TrajectoryState) -> List[RerankerAgentData]:
        """Construct reranker agent data from trajectory.
        
        Args:
            trajectory: TrajectoryState object containing messages and executed_tool_calls.

        Returns:
            List of RerankerAgentData, one for each tool call position.
        """
        assert validate_mssg_tool_call_match(trajectory), "Messages and tool calls do not match!"

        if trajectory.is_valid is False:
            return []  # Skip invalid trajectories
        
        messages = trajectory.messages
        tool_calls = trajectory.executed_tool_calls
        
        # Find all tool call positions (assistant followed by tool)
        tool_call_positions = [
            i for i in range(len(messages) - 1)
            if messages[i]["role"] == "assistant" and messages[i + 1]["role"] == "tool"
        ]
        
        # Create agent data for each position, with n rollouts per position
        # All n rollouts for the same position share the same uid
        n_rollouts = self.config.reranker_actor_rollout_ref.rollout.n
        agent_data_list = []
        for idx, pos in enumerate(tool_call_positions):
            shared_uid = str(uuid.uuid4())  # Same uid for all n rollouts at this position
            for _ in range(n_rollouts):
                agent_data_list.append(
                    RerankerAgentData(
                        uid=shared_uid,
                        prefix_messages=messages[:pos + 1],  # Include the assistant that triggered the tool
                        tool_call=tool_calls[idx],
                        user_turns=sum(1 for m in messages[:pos + 1] if m["role"] == "tool"),
                        assistant_turns=sum(1 for m in messages[:pos + 1] if m["role"] == "assistant"),
                        reward_kwargs=trajectory.reward_kwargs,  # Pass reward kwargs from trajectory
                    )
                )
        
        return agent_data_list 
    
    async def _call_reranker_tool_as_agent(self, reranker_agent_data: RerankerAgentData) -> RerankerAgentData:
        """Call reranker as agent mode to generate n different responses.
        
        Args:
            reranker_agent_data: RerankerAgentData containing tool_call and prefix_messages.
            
        Returns:
            Updated RerankerAgentData with response_ids, response_mask, response_logprobs.
        """
        search_r1_agent_loop = SearchR1AgentLoop(
            trainer_config=_DummyConfig(self.config),
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            reranker_server_manager=self.reranker_server_manager,
            reranker_tokenizer=self.reranker_tokenizer,
            track_messages=True,
        )
        
        # Call tool as agent using SearchR1AgentLoop instance
        answers = reranker_agent_data.reward_kwargs["reward_model"]["ground_truth"]["target"]
        agent_tool_response, tool_reward, tool_metric = await search_r1_agent_loop.call_tool_as_agent(reranker_agent_data.tool_call,
                                                                                            answers=answers)

        if tool_metric["reranker_crashed"]:
            reranker_agent_data.reranker_crashed = True
            return reranker_agent_data

        # not error when calling reranker, only reranker output don't follow required format
        if tool_metric["reranker_success"]:
            reranker_agent_data.is_success = True
        else:
            tool_reward = reranker_agent_data.format_penalty
            reranker_agent_data.is_success = False

        # Update reranker_agent_data with response from reranker
        reranker_agent_data.response_ids = agent_tool_response.response_ids
        reranker_agent_data.response_mask = agent_tool_response.response_mask
        reranker_agent_data.response_logprobs = agent_tool_response.response_logprobs
        reranker_agent_data.prompt_ids = agent_tool_response.prompt_ids
        
        # Store tool metrics for tracking reranker success/fallback
        reranker_agent_data.tool_metrics = tool_metric
        reranker_agent_data.tool_reward = tool_reward

        messages = reranker_agent_data.prefix_messages + [{"role": "tool", "content": agent_tool_response.text}]
        reranker_agent_data.raw_messages = messages         

        return reranker_agent_data

    async def _record_search_r1_continuation_info(
        self, 
        reranker_data: RerankerAgentData, 
        agent_output: Optional[_InternalAgentLoopOutput]
    ) -> dict[str, Any]:
        """Record search-r1 continuation information after reranker generates new observation.
        
        Args:
            reranker_data: RerankerAgentData with prefix messages.
            agent_output: Optional agent output from search-r1 continuation.
            
        Returns:
            Dictionary with search-r1 continuation info (prompt, response, prefix_messages).
        """
        loop = asyncio.get_running_loop()
        
        if agent_output is not None:
            # Decode in parallel using executor to avoid blocking
            input_decode_task = loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(agent_output.input_ids[0], skip_special_tokens=True)
            )
            response_decode_task = loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(agent_output.response_ids[0], skip_special_tokens=True)
            )
            
            input_text, response_text = await asyncio.gather(input_decode_task, response_decode_task)
            
            return {
                "prefix_messages": reranker_data.raw_messages,
                "input": input_text,
                "response": response_text,
            }
        else:
            return {
                "prefix_messages": reranker_data.raw_messages,
                "input": "none",
                "response": "none",
            }

    def _calculate_reranker_target_size(self, main_train_size: int, has_valid_groups: bool) -> int:
        """Calculate target size for reranker training samples with FSDP alignment.
        
        Args:
            main_train_size: Number of main agent training samples.
            has_valid_groups: Whether there are valid groups (with reward variance).
            
        Returns:
            Target size aligned to FSDP requirements.
        """
        rollout_n = self.config.reranker_actor_rollout_ref.rollout.n
        
        if not has_valid_groups:
            # When no valid groups, use minimal valid size
            if rollout_n % 4 != 0:
                return rollout_n * self.reranker_n_gpus
            else:
                return rollout_n
        
        # Normal case: cap based on main_train_size ratio
        target_size = int(main_train_size * self.reranker_main_train_n_ratio)
        
        # Align to FSDP requirements if rollout_n not divisible by 4
        if rollout_n % 4 != 0:
            multiple = rollout_n * self.reranker_n_gpus
            target_size = (target_size // multiple + 1) * multiple
        
        return target_size
    
    def _sample_groups_to_target(self, group_list: list, target_size: int) -> list:
        """Sample groups to reach target size, with upsampling if needed.
        
        Args:
            group_list: List of groups (each group has rollout_n samples with same uid).
            target_size: Target number of samples.
            
        Returns:
            Flattened list of samples reaching target size.
        """
        rollout_n = self.config.reranker_actor_rollout_ref.rollout.n
        available_samples = len(group_list) * rollout_n
        
        if available_samples >= target_size:
            # Cap to target_size
            filtered_list = []
            random.shuffle(group_list)
            for group in group_list:
                filtered_list.extend(group)
                if len(filtered_list) >= target_size:
                    break
            return filtered_list
        else:
            # Need to upsample
            if rollout_n % 4 != 0:
                # Align to (rollout_n * 8) multiple
                multiple = rollout_n * self.reranker_n_gpus
                aligned_size = ((available_samples // multiple) + 1) * multiple
            else:
                # Use available samples directly (already aligned)
                aligned_size = available_samples
            
            filtered_list = []
            while len(filtered_list) < aligned_size:
                for group in group_list:
                    filtered_list.extend(group)
                    if len(filtered_list) >= aligned_size:
                        break
            
            return filtered_list
    
    def _filter_all_same_rewards_and_cap_size(self, 
                                              list_rank_data: list[_InternalAgentLoopOutput], 
                                              main_train_size: int) -> list[_InternalAgentLoopOutput]:
        """Filter out samples where all reranker branches have the same reward.
        
        For GRPO training:
        1. Filter out groups where all samples with same uid have identical rewards
        2. Sample/upsample to reach target size aligned to FSDP requirements
        
        Args:
            list_rank_data: List of _InternalAgentLoopOutput with rewards.
            main_train_size: Number of main agent training samples to cap reranker size.
            
        Returns:
            Filtered list of _InternalAgentLoopOutput aligned to FSDP requirements.
        """
        if not list_rank_data:
            return []
        
        # Group by uid
        from collections import defaultdict
        uid_groups = defaultdict(list)
        for data in list_rank_data:
            uid_groups[data.extra_fields["uid"]].append(data)
        
        # Filter out groups where all rewards are the same OR all rerank outputs are the same
        valid_groups = {}
        rollout_n = self.config.reranker_actor_rollout_ref.rollout.n
        
        for uid, group in uid_groups.items():
            # Check 0: Some groups might have fewer than n_rollouts samples due to crashed samples
            # Upsample by randomly copying existing samples to reach n_rollouts
            if len(group) < rollout_n:
                num_to_add = rollout_n - len(group)
                logger.info(
                    f"Group {uid} has only {len(group)}/{rollout_n} samples (some crashed). "
                    f"Upsampling by copying {num_to_add} samples."
                )
                group.extend(random.choices(group, k=num_to_add))

            # Check 1: Filter if all rewards are identical
            rewards = [data.reward_score for data in group]
            if len(set(rewards)) == 1:
                logger.info(f"Filtering out uid {uid}: all {len(rewards)} samples have same reward {rewards[0]}")
                continue
            
            # Check 2: Filter if all rerank outputs are identical
            rerank_outputs = []
            for data in group:
                # Decode response_ids to get output text
                response_text = self.reranker_tokenizer.decode(data.response_ids[0], skip_special_tokens=True)
                # Extract <rerank>...</rerank> content
                rerank = extract_rerank_output(response_text)
                if rerank:
                    rerank_outputs.append(rerank)
            
            # If we have rerank outputs and they're all identical, filter out
            if len(rerank_outputs) > 0 and len(set(rerank_outputs)) == 1:
                logger.info(f"Filtering out uid {uid}: all {len(rerank_outputs)} samples have identical rerank output: {rerank_outputs[0]}")
                continue
            
            # Passed both checks
            valid_groups[uid] = group
        
        # Determine which groups to use
        has_valid_groups = len(valid_groups) > 0
        if has_valid_groups:
            group_list = list(valid_groups.values())
        else:
            logger.warning(
                "All uid groups filtered out due to identical rewards! "
                "Using all groups (including filtered ones) for training."
            )
            group_list = list(uid_groups.values())
        
        # Calculate target size
        target_size = self._calculate_reranker_target_size(main_train_size, has_valid_groups)
        
        # Sample groups to reach target size
        filtered_list = self._sample_groups_to_target(group_list, target_size)
        
        print(f"Reward filtering and cap_size: {len(list_rank_data)} -> {len(filtered_list)} samples (target={target_size})")
        assert len(filtered_list) % self.reranker_n_gpus == 0, (
            f"Filtered reranker samples must be multiple of {self.reranker_n_gpus} for FSDP training! "
            f"Got {len(filtered_list)} samples (target={target_size})"
        )
        
        return filtered_list

    async def generate_sequences_counterfactual(self, batch: DataProto) -> tuple[DataProto, DataProto]:
        """Generate sequences with counterfactual rollout for reranker training.
        
        For each prompt:
        1. Run initial rollout to get full trajectory
        2. Identify branch points (tool call positions)
        3. Create branches at each point (4 samples per branch point)
        4. Continue generation from each branch
        5. filter invalid samples 
        
        Args:
            batch: Input batch of prompts.
            
        Returns:
            Output batch with counterfactual branches and GRPO grouping.
        """    
        # Step 1: Run initial rollouts
        start_time = time.time()
        print("[generate_sequences_counterfactual] Step 1: batch size =", len(batch))
        main_agent_output, initial_trajectories = await self.generate_search_r1_initial_sequences(batch)
        print("[generate_sequences_counterfactual] Completed initial rollouts in", time.time() - start_time, "seconds")

        # Step 2: Construct branches from trajectories
        reranker_agent_data_list = []
        for traj in initial_trajectories:
            agent_data_list = self._construct_reranker_train_data(traj)
            reranker_agent_data_list.extend(agent_data_list)

        # Step 3: Process reranker pipeline (reranker call + search-r1 continuation) with GPU pipelining
        # By processing each data point through the complete pipeline, we can utilize all 16 GPUs:
        # - Some tasks will be calling reranker (using reranker 8 GPUs)
        # - Other tasks will be calling search-r1 continuation (using main agent 8 GPUs)
        # This is much more efficient than batching all reranker calls first, then all continuations
        start_time = time.time()
        print("[generate_sequences_counterfactual] Step 2+3: processing", len(reranker_agent_data_list), "data points through complete pipeline")
        pipeline_tasks = [
            asyncio.create_task(self._call_with_semaphore(self._process_reranker_pipeline, data))
            for data in reranker_agent_data_list
        ]
        outputs = await asyncio.gather(*pipeline_tasks)
        print("[generate_sequences_counterfactual] Completed pipeline for", len(outputs), "trajectories in", time.time() - start_time, "seconds")
        
        # Filter out samples with execution_error (infra issues) - these should not be used for training
        filtered_outputs = []
        num_crashed = 0  # Reranker crashed before completion
        num_execution_errors = 0  # Reranker completed but with execution error
        num_format_errors = 0
        num_success = 0
        
        for output in outputs:
            # Handle None output (reranker crashed)
            if output is None:
                num_crashed += 1
                continue
            
            # Handle normal output with metrics
            if output.extra_fields:
                fallback_reason = output.extra_fields.get("reranker_fallback_reason")
                if fallback_reason == "execution_error":
                    num_execution_errors += 1
                    logger.warning(f"Filtering out sample due to reranker execution_error (infra issue)")
                    continue  # Skip this sample
                elif fallback_reason == "format_validation_error":
                    num_format_errors += 1
                elif output.extra_fields.get("reranker_success"):
                    num_success += 1
            
            filtered_outputs.append(output)

        print(
            f"Reranker stats: {num_success} success, {num_format_errors} format_errors (penalty=-0.2), "
            f"{num_crashed} crashed (filtered), {num_execution_errors} execution_errors (filtered)"
        )

        if not filtered_outputs:
            logger.error("All samples filtered out due to execution errors!")
            raise RuntimeError("No valid samples remaining after filtering execution errors")

        # more filters based on reward
        filtered_outputs = self._filter_all_same_rewards_and_cap_size(filtered_outputs, len(main_agent_output))

        
        reranker_output = self._postprocess(filtered_outputs)

        return (main_agent_output, reranker_output)

    
    async def _continue_search_r1_from_reranker_agent_loop(self, reranker_data: RerankerAgentData) -> _InternalAgentLoopOutput:
        """Continue generation from reranker's tool response to compute reward.
        Then build dataproto for reranker training.
        
        Args:
            reranker_data: RerankerAgentData with raw_messages (prefix + tool response).
            
        Returns:
            _InternalAgentLoopOutput updated with reward_score.
        """
        # Create a temporary batch for this single trajectory
        # Use the messages from reranker (prefix + tool response)
        search_r1_config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=search_r1_config.val_kwargs.temperature,
            top_p=search_r1_config.val_kwargs.top_p,
            repetition_penalty=1.0,
            logprobs=False,  # Don't need logprobs for evaluation
        )
        
        # Create trajectory info
        trajectory_info = {
            "step": -1,
            "sample_index": 0,
            "rollout_n": 0,
            "validate": False,
        }
        
        # Call agent loop with reranker's messages and reward kwargs
        # Use search-r1 agent to continue from reranker's observation
        kwargs = {
            "raw_prompt": copy.deepcopy(reranker_data.raw_messages),
            "agent_name": search_r1_config.agent.default_agent_loop,  # Use search-r1 agent, not reranker
            # Add reward computation kwargs from original data
            **reranker_data.reward_kwargs,
        }
        
        # Determine final reward based on tool metrics
        tool_metrics = reranker_data.tool_metrics
        fallback_reason = tool_metrics["reranker_fallback_reason"]

        # Run search-r1 agent
        num_turns = 0
        agent_output = None
        if reranker_data.is_success:
            agent_output = await self._run_agent_loop(
                sampling_params=sampling_params,
                trajectory=trajectory_info,
                trace=False,
                **kwargs
            )
            agent_reward = agent_output.reward_score
            num_turns = agent_output.num_turns
        else:
            if fallback_reason == "format_validation_error":
                agent_reward = reranker_data.tool_reward  # Use tool reward (with -0.2 penalty)
                logger.warning(f"Reranker format validation failed, applying penalty: -0.2")
            elif fallback_reason == "execution_error":
                agent_reward = -1.0  # Special marker for filtering
                logger.error(f"Reranker execution error (infra issue), marking sample as invalid")
            else:
                raise ValueError(f"Unexpected reranker state: success={reranker_data.is_success}, reason={fallback_reason}")
            
        # we compute the reward_score based on binary agent reward
        final_reward = self.aggregate_tool_and_agent_reward(
            tool_reward=reranker_data.tool_reward,
            agent_reward=agent_reward,
            is_reranker_success=reranker_data.is_success,
        )
        reranker_data.reward_score = final_reward

        # start to build proto data
        self.reranker_tokenizer.padding_side = "left"
        prompt_output = self.reranker_tokenizer.pad(
            {"input_ids": reranker_data.prompt_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.reranker_tokenizer.padding_side = "right"
        response_output = self.reranker_tokenizer.pad(
            {"input_ids": reranker_data.response_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.reranker_tokenizer.pad(
            {"input_ids": reranker_data.response_mask},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if reranker_data.response_logprobs is not None:
            pad_size = self.config.actor_rollout_ref.rollout.response_length - len(reranker_data.response_logprobs)
            response_logprobs = torch.tensor(reranker_data.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        # Prepare reranker metrics for wandb tracking
        # Ensure all samples have the same keys for DataProto.concat
        reranker_metrics = {
            "reranker_attempted": tool_metrics.get("reranker_attempted", False),
            "reranker_success": tool_metrics.get("reranker_success", False),
            "reranker_fallback": tool_metrics.get("reranker_fallback", False),
            "reranker_fallback_reason": tool_metrics.get("reranker_fallback_reason"),
            "num_retrieved_docs": tool_metrics.get("num_retrieved_docs", 0),
            "num_reranked_docs": tool_metrics.get("num_reranked_docs", 0),
            "reranker_error_type": tool_metrics.get("reranker_error_type", None),
            "reranker_validation_errors": str(tool_metrics["reranker_validation_errors"])[:200] if "reranker_validation_errors" in tool_metrics else None,
        }
        
        # Build extra_fields - if we have agent_output, use its extra_fields and add reranker metrics
        # Otherwise, create minimal extra_fields with required keys
        if agent_output is not None:
            extra_fields = agent_output.extra_fields.copy()
            # Ensure reward_extra_info has all required keys for consistency across all samples
            if "reward_extra_info" in extra_fields:
                reward_info = extra_fields["reward_extra_info"]
                if "json_correct" not in reward_info:
                    reward_info["json_correct"] = []
                if "one_tool_call_per_assistant" not in reward_info:
                    reward_info["one_tool_call_per_assistant"] = []
            extra_fields.update(reranker_metrics)
            extra_fields["reward_extra_info"]["agent_reward"] = agent_reward
            extra_fields["reward_extra_info"]["tool_reward"] = reranker_data.tool_reward
        else:
            # Create minimal extra_fields matching the structure from normal agent loop
            extra_fields = {
                "messages": reranker_data.raw_messages,
                "executed_tool_calls": [],
                "turn_scores": [],
                "tool_rewards": [],
                "json_correct": [],
                "one_tool_call_per_assistant": [],
                # Hardcode reward_extra_info with same keys as successful case
                "reward_extra_info": {
                    "score": final_reward,  # -0.2 or -1.0
                    "valid": False,
                    "f1": 0.0,
                    "json_correct": [],  # Match successful case structure
                    "one_tool_call_per_assistant": [],  # Match successful case structure
                },
            }
            extra_fields.update(reranker_metrics)
            extra_fields["reward_extra_info"]["agent_reward"] = agent_reward
            extra_fields["reward_extra_info"]["tool_reward"] = reranker_data.tool_reward

        # add uid back
        extra_fields["uid"] = reranker_data.uid

        # preserve reward_model for logging ground truth
        extra_fields["reward_model"] = reranker_data.reward_kwargs.get("reward_model", None)

        # Record search-r1 continuation info (async decode to avoid blocking)
        extra_fields["search_r1"] = await self._record_search_r1_continuation_info(reranker_data, agent_output)
        
        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            reward_score=reranker_data.reward_score,
            num_turns=num_turns,
            metrics={},
            extra_fields=extra_fields,
        )

    def aggregate_tool_and_agent_reward(self, tool_reward: float, 
                                        agent_reward: float, 
                                        is_reranker_success: bool,
                                        **kwargs) -> float:
        """Aggregate tool reward and agent reward into final reward score.
        
        Args:
            tool_reward: Reward from tool (reranker).
            agent_reward: Reward from agent continuation.

        Returns:
            Final aggregated reward score.
        """ 
        if not is_reranker_success:
            assert tool_reward == -0.2, "Format penalty should be -0.2 when reranker not success"
            return tool_reward

        # only F1 score > 0.8, we think search-r1 has the correct answer
        binary_agent_reward = 1.0 if agent_reward >= 0.8 else 0.0
        return max(tool_reward, binary_agent_reward)
    
    def _cap_reranker_output(self, reranker_output: list[_InternalAgentLoopOutput], max_size: int) -> list[_InternalAgentLoopOutput]:
        """Cap reranker output size to max_size by random sampling.
        
        Args:
            reranker_output: Original reranker DataProto.
            max_size: Maximum allowed size.
            
        Returns:
            Capped reranker DataProto.
        """
        if len(reranker_output) <= max_size:
            return reranker_output
        
        return random.sample(reranker_output, max_size)