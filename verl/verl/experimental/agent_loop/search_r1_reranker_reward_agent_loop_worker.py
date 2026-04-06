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
import json
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
    SearchR1RerankerAgentLoopWorkerBase,
    SearchR1RerankerRewardAgentLoopWorkerBase,
    _InternalAgentLoopOutput,
    _DummyConfig,
)
from verl.experimental.agent_loop.counterfactual_rollout import (
    TrajectoryState,
)
from verl.experimental.agent_loop.function_loaders import load_custom_function
from verl.experimental.agent_loop.uid_group_functions import group_by_muid_ans_in_doc
from verl.experimental.agent_loop.score_assign_functions import max_tool_agent_score
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
class SearchR1RerankerRewardAgentLoopWorker(SearchR1RerankerRewardAgentLoopWorkerBase):
    """Extended agent loop worker with counterfactual rollout support.
    
    This worker extends DualAgentLoopWorker to add counterfactual rollout capabilities
    for training the reranker agent with GRPO.
    
    Supports two reward judge modes:
    - Ray mode: Reward judge as Ray-managed vLLM server (2-node architecture)
    - HTTP mode: Reward judge as external HTTP server (3-node architecture)
    """
    
    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
        reranker_server_handles: list[ray.actor.ActorHandle] = None,
        reward_server_handles: list[ray.actor.ActorHandle] = None,
        reward_http_server_urls: list[str] = None,
    ):
        """Initialize extended agent loop worker.
        
        Args:
            config: YAML config.
            server_handles: Main agent server actor handles.
            reward_router_address: Reward router address.
            reranker_server_handles: Optional reranker server actor handles.
            reward_server_handles: Reward model (LLM-as-Judge) server handles (Ray mode).
            reward_http_server_urls: List of external reward judge server URLs (HTTP mode).
        """
        # Use keyword args to avoid positional order confusion between
        # worker (reward_router_address 3rd) and parent base class
        super().__init__(
            config=config,
            server_handles=server_handles,
            reranker_server_handles=reranker_server_handles,
            reward_router_address=reward_router_address,
            reward_server_handles=reward_server_handles,
            reward_http_server_urls=reward_http_server_urls,
        )

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
        
        # Load configurable UID grouping function
        self.uid_group_fn = load_custom_function(
            config,
            config_key="reranker_uid_group_function",
            kwargs_key="uid_group_kwargs",
            default_fn=group_by_muid_ans_in_doc,
        )
        logger.info(f"Loaded UID grouping function: {self.uid_group_fn}")
        
        # Load configurable score assignment function
        self.score_assign_fn = load_custom_function(
            config,
            config_key="reranker_score_assign_function",
            kwargs_key="score_assign_kwargs",
            default_fn=max_tool_agent_score,
        )
        logger.info(f"Loaded score assignment function: {self.score_assign_fn}")
        
        # Read n_judge_samples for padding raw_scores/raw_texts to fixed length
        judge_config = config.get("reward_judge_model", {})
        self.n_judge_samples = judge_config.get("n_judge_samples", 4)
        
    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> tuple[DataProto, list, Optional[DataProto]]:
        """Generate sequences from agent loop.
        The code is copied from generate_sequences in AgentLoopWorkerBase that remove the postprocess step.

        Args:
            batch (DataProto): Input batch.

        Returns:
            Tuple of (main_agent_batch, trajectories, reranker_batch):
            - main_agent_batch: DataProto for main agent outputs
            - trajectories: List of trajectory info for constructing reranker train data
            - reranker_batch: DataProto for reranker outputs (or None if empty)
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
        is_validate = batch.meta_info.get("validate", False)
        if is_validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature
        
        # Prepare reranker sampling params based on training/validate mode
        reranker_config = self.config.reranker_actor_rollout_ref.rollout
        global_steps = batch.meta_info.get("global_steps", -1)
        val_start_step = self.config.trainer.get("reranker_sampling_val_start_step", -1)
        use_val_params_for_reranker = is_validate or (
            val_start_step is not None and val_start_step >= 0 and global_steps >= val_start_step
        )
        if use_val_params_for_reranker:
            print("[in AgentLoopWorker] Using validation sampling params for reranker in training.")
            reranker_sampling_params = {
                "temperature": reranker_config.val_kwargs.temperature,
                "top_p": reranker_config.val_kwargs.top_p,
            }
        else:
            print("[in AgentLoopWorker] Using training sampling params for reranker in training.")
            reranker_sampling_params = {
                "temperature": reranker_config.temperature,
                "top_p": reranker_config.top_p,
            }

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
            
            # Inject reranker_sampling_params into tools_kwargs for each sample
            if "tools_kwargs" not in kwargs or kwargs["tools_kwargs"] is None:
                kwargs["tools_kwargs"] = {}
            if "search" not in kwargs["tools_kwargs"]:
                kwargs["tools_kwargs"]["search"] = {}
            if "create_kwargs" not in kwargs["tools_kwargs"]["search"]:
                kwargs["tools_kwargs"]["search"]["create_kwargs"] = {}
            kwargs["tools_kwargs"]["search"]["create_kwargs"]["reranker_sampling_params"] = reranker_sampling_params
            
            kwargs_list.append(kwargs)
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)
        
        # Unpack tuple outputs: (main_output, reranker_outputs)
        main_outputs = [output[0] for output in outputs]
        reranker_outputs_list = [output[1] for output in outputs]
        
        main_agent_batch = self._postprocess(main_outputs)

        if is_validate:
            return main_agent_batch

        # add answers and initial_question to main_agent_batch
        assert len(main_outputs) == len(batch), "Output length mismatch!"

        # Process reranker outputs: assign uid, compute final_score, filter, cap size
        # Convert List[List[AgentLoopOutput]] to List[_InternalAgentLoopOutput]
        reranker_internal_outputs, reranker_metrics = await self._process_reranker_outputs(
            reranker_outputs_list, len(batch)
        )
        
        reranker_batch = self._postprocess(reranker_internal_outputs)
        # Return metrics separately (don't put in meta_info to avoid concat conflicts)
        reranker_batch.meta_info["worker_reranker_metrics"] = reranker_metrics
        
        return main_agent_batch, reranker_batch
        
    async def _process_reranker_outputs(
        self, 
        reranker_outputs_list: list[list[AgentLoopOutput]], 
        main_batch_size: int
    ) -> tuple[list[_InternalAgentLoopOutput], dict]:
        """Process reranker outputs: assign uid, filter, cap size, collect metrics.
        
        Args:
            reranker_outputs_list: List of lists of AgentLoopOutput from each sample.
            main_batch_size: Size of main agent batch (for cap calculation).
            
        Returns:
            Tuple of (processed internal outputs, metrics dict).
        """
        # Step 1: Flatten and add trajectory_id and step_index
        flat_outputs = []
        for traj_idx, outputs_per_sample in enumerate(reranker_outputs_list):
            trajectory_id = f"traj_{traj_idx}"
            for step_idx, output in enumerate(outputs_per_sample):
                output.extra_fields["trajectory_id"] = trajectory_id
                output.extra_fields["step_index"] = step_idx
                flat_outputs.append(output)
        
        if not flat_outputs:
            return [], {}
        
        # Step 2: Assign UID and compute final_score (per-output)
        self._assign_uid_and_final_score(flat_outputs)
        
        # Collect pre-filter metrics
        pre_filter_metrics = self._collect_pre_filter_metrics(flat_outputs)
        
        # Step 3: Reassign rewards for same prompt+ranking with different scores
        duplicate_metrics = self._reassign_rewards_for_duplicates(flat_outputs)
        
        # Step 4: Group by UID
        uid_groups = self._group_by_uid(flat_outputs)
        
        # Step 5: Filter groups (all same rewards, group size <= 2)
        filtered_groups, filter_metrics = self._filter_groups(uid_groups)
        
        # Step 6: Cap size to target and align to reranker_n_gpus
        target_size = self.reranker_main_train_n_ratio * main_batch_size  # self.reranker_main_train_n_ratio = 2
        final_outputs, cap_metrics = self._cap_and_align_outputs(
            filtered_groups, target_size, self.reranker_n_gpus
        )
        
        # Step 6.5: Fallback if all outputs were filtered
        if not final_outputs:
            logger.warning("All outputs filtered! Using fallback to keep outputs.")
            final_outputs = self._fallback_keep_outputs_to_align(uid_groups, self.reranker_n_gpus)
            cap_metrics["fallback/triggered"] = 1
            cap_metrics["fallback/kept_outputs"] = len(final_outputs)
        else:
            cap_metrics["fallback/triggered"] = 0
        
        # Step 7: Convert to internal format (sequential to avoid tokenizer race condition)
        internal_outputs = []
        for output in final_outputs:
            internal_output = self._convert_reranker_agent_output_to_internal(
                output,
                self.reranker_tokenizer,
                self.config.reranker_actor_rollout_ref.rollout.prompt_length,
                self.config.reranker_actor_rollout_ref.rollout.response_length
            )
            internal_outputs.append(internal_output)
        
        # Step 8: Merge all metrics
        all_metrics = {
            **pre_filter_metrics,
            **duplicate_metrics,
            **filter_metrics,
            **cap_metrics,
        }
        
        return internal_outputs, all_metrics
    
    def _assign_uid_and_final_score(self, outputs: list[AgentLoopOutput]) -> None:
        """Assign UID and compute final_score using configurable functions.
        
        This method uses self.uid_group_fn and self.score_assign_fn which are loaded
        from config in __init__. These functions can be customized via YAML config.
        
        Modifies outputs in-place:
        - Sets extra_fields["uid"] using configured uid_group_fn
        - Sets extra_fields["final_score"] using configured score_assign_fn
        - Sets extra_fields["reward_extra_info"] with reranker-specific scores for logging
        """
        # Call the configurable UID grouping function
        # It modifies outputs in-place by setting extra_fields["uid"]
        self.uid_group_fn(outputs)
        
        # Compute final_score for each output using configurable score assignment function
        for output in outputs:
            tool_score = output.extra_fields.get("tool_score", 0.0)
            agent_score = output.extra_fields.get("agent_score", 0.0)
            answer_in_docs = output.extra_fields.get("answer_in_docs", False)
            
            # Extract LLM judge score (may be None if judge failed or not enabled)
            llm_judge_score_raw = output.extra_fields.get("llm_judge_score", None)
            # Fallback: if judge failed, use tool_score if non-negative, else 0.0
            if llm_judge_score_raw is None:
                llm_judge_score = max(tool_score, 0.0) if tool_score >= 0 else 0.0
            else:
                llm_judge_score = llm_judge_score_raw
            
            # Call the configurable score assignment function
            final_score = self.score_assign_fn(
                tool_score=tool_score,
                agent_score=agent_score,
                answer_in_docs=answer_in_docs,
                llm_judge_score=llm_judge_score,
                output=output,  # Pass full output for context-aware scoring
            )
            output.extra_fields["final_score"] = final_score
    
            # Pad raw_scores and raw_texts to fixed n_judge_samples length
            # so numpy can stack them into uniform arrays in _postprocess.
            n = self.n_judge_samples
            raw_scores = output.extra_fields.get("llm_judge_raw_scores", [])
            raw_texts = output.extra_fields.get("llm_judge_raw_texts", [])
            padded_scores = (raw_scores + [0.0] * n)[:n]
            padded_texts = (raw_texts + [""] * n)[:n]
            
            output.extra_fields["reward_extra_info"] = {
                "final_score": final_score,
                "tool_score": tool_score,
                "agent_score": agent_score,
                "llm_judge_score": llm_judge_score,
                "llm_judge_score_raw": llm_judge_score_raw if llm_judge_score_raw is not None else 0.0,
                "llm_judge_n_success": len(raw_scores),
                "llm_judge_raw_scores": padded_scores,
                "llm_judge_raw_texts": padded_texts,
                "answer_in_docs": answer_in_docs,
                "golden_answer": output.extra_fields.get("golden_answer", []),
                "sub_query": output.extra_fields.get("sub_query", ""),
            }
    
    def _collect_pre_filter_metrics(self, outputs: list[AgentLoopOutput]) -> dict:
        """Collect metrics before filtering."""
        if not outputs:
            return {}
        
        # Collect scores
        final_scores = [o.extra_fields.get("final_score", 0.0) for o in outputs]
        tool_scores = [o.extra_fields.get("tool_score", 0.0) for o in outputs]
        agent_scores = [o.extra_fields.get("agent_score", 0.0) for o in outputs]
        
        # Group sizes
        uid_groups = self._group_by_uid(outputs)
        group_sizes = [len(g) for g in uid_groups.values()]
        
        # LLM-as-Judge metrics
        llm_judge_metrics = {}
        llm_judge_scores = []
        llm_judge_raw_score_lists = []  # per-sample variance tracking
        llm_judge_success = 0
        llm_judge_total = 0
        judge_when_ans_in_docs = []
        judge_when_ans_not_in_docs = []
        
        for output in outputs:
            judge_score = output.extra_fields.get("llm_judge_score", None)
            raw_scores = output.extra_fields.get("llm_judge_raw_scores", [])
            ans_in_docs = output.extra_fields.get("answer_in_docs", False)
            
            if judge_score is not None:
                llm_judge_scores.append(judge_score)
                llm_judge_success += 1
                if ans_in_docs:
                    judge_when_ans_in_docs.append(judge_score)
                else:
                    judge_when_ans_not_in_docs.append(judge_score)
            llm_judge_total += 1
            
            if len(raw_scores) > 1:
                llm_judge_raw_score_lists.append(raw_scores)
        
        if llm_judge_scores:
            llm_judge_metrics["llm_judge/avg_score"] = np.mean(llm_judge_scores)
            llm_judge_metrics["llm_judge/score_std"] = np.std(llm_judge_scores)
        else:
            llm_judge_metrics["llm_judge/avg_score"] = 0.0
            llm_judge_metrics["llm_judge/score_std"] = 0.0
        
        llm_judge_metrics["llm_judge/avg_success_rate"] = llm_judge_success / max(llm_judge_total, 1)
        llm_judge_metrics["llm_judge/total_calls"] = llm_judge_total
        
        # Per-sample standard deviation (measures agreement across N judge calls)
        if llm_judge_raw_score_lists:
            per_sample_stds = [np.std(scores) for scores in llm_judge_raw_score_lists]
            llm_judge_metrics["llm_judge/avg_per_sample_std"] = np.mean(per_sample_stds)
        else:
            llm_judge_metrics["llm_judge/avg_per_sample_std"] = 0.0
        
        # Score conditioned on answer_in_docs
        llm_judge_metrics["llm_judge/avg_when_ans_in_docs"] = (
            np.mean(judge_when_ans_in_docs) if judge_when_ans_in_docs else 0.0
        )
        llm_judge_metrics["llm_judge/avg_when_ans_not_in_docs"] = (
            np.mean(judge_when_ans_not_in_docs) if judge_when_ans_not_in_docs else 0.0
        )
        llm_judge_metrics["llm_judge/count_ans_in_docs"] = len(judge_when_ans_in_docs)
        llm_judge_metrics["llm_judge/count_ans_not_in_docs"] = len(judge_when_ans_not_in_docs)
        
        return {
            "pre_filter/total_outputs": len(outputs),
            "pre_filter/num_groups": len(uid_groups),
            "pre_filter/avg_final_score": np.mean(final_scores),
            "pre_filter/avg_tool_score": np.mean(tool_scores),
            "pre_filter/avg_agent_score": np.mean(agent_scores),
            "pre_filter/avg_group_size": np.mean(group_sizes),
            "pre_filter/min_group_size": np.min(group_sizes),
            "pre_filter/max_group_size": np.max(group_sizes),
            "pre_filter/ratio_valid": np.mean([s >= 0.0 for s in tool_scores]),
            **llm_judge_metrics,
        }
    
    def _reassign_rewards_for_duplicates(self, outputs: list[AgentLoopOutput]) -> dict:
        """Reassign final_score for same prompt+ranking with different scores.
        
        Uses tuple of token IDs as key for guaranteed correctness (no hash collision).
        Time complexity: O(n) using hash map.
        
        Returns:
            Dict with duplicate statistics.
        """
        # Group by UID first
        uid_groups = self._group_by_uid(outputs)
        
        total_outputs = len(outputs)
        total_duplicates = 0
        num_duplicate_groups = 0
        
        for uid, group in uid_groups.items():
            # Build hash map: (prompt_tuple, ranking_str) -> list of outputs
            duplicate_map = {}
            
            for output in group:
                # Use tuple of token IDs directly as key
                # Python dict will hash internally and compare actual values on collision
                prompt_key = tuple(output.prompt_ids)
                
                # For ranking, extract the content directly (it's short)
                response_str = self.reranker_tokenizer.decode(output.response_ids, skip_special_tokens=True)
                ranking_str = extract_rerank_output(response_str)
                
                key = (prompt_key, ranking_str if ranking_str else "")
                if key not in duplicate_map:
                    duplicate_map[key] = []
                duplicate_map[key].append(output)
            
            # For each group with duplicates, assign average final_score
            for key, dup_outputs in duplicate_map.items():
                if len(dup_outputs) > 1:
                    # Same prompt+ranking but different scores - average them
                    avg_score = np.mean([o.extra_fields["final_score"] for o in dup_outputs])
                    for output in dup_outputs:
                        output.extra_fields["final_score"] = avg_score
                        output.extra_fields["reward_extra_info"]["final_score"] = avg_score
                    
                    # Count duplicates (all but one are duplicates)
                    total_duplicates += len(dup_outputs) - 1
                    num_duplicate_groups += 1
        
        return {
            "duplicate/total_outputs": total_outputs,
            "duplicate/num_duplicates": total_duplicates,
            "duplicate/num_duplicate_groups": num_duplicate_groups,
            "duplicate/ratio": total_duplicates / total_outputs if total_outputs > 0 else 0.0,
        }
    
    def _group_by_uid(self, outputs: list[AgentLoopOutput]) -> dict[str, list[AgentLoopOutput]]:
        """Group outputs by UID."""
        from collections import defaultdict
        uid_groups = defaultdict(list)
        for output in outputs:
            uid = output.extra_fields["uid"]
            uid_groups[uid].append(output)
        return dict(uid_groups)
    
    def _filter_groups(
        self, 
        uid_groups: dict[str, list[AgentLoopOutput]]
    ) -> tuple[dict[str, list[AgentLoopOutput]], dict]:
        """Filter groups: remove groups with all same final_score or size <= 2."""
        filtered_groups = {}
        num_filtered_same_score = 0
        num_filtered_small_size = 0
        
        for uid, group in uid_groups.items():
            # Filter condition 1: group size <= 2
            if len(group) <= 2:
                num_filtered_small_size += 1
                continue
            
            # Filter condition 2: all final_scores are the same
            final_scores = [o.extra_fields.get("final_score", 0.0) for o in group]
            if len(set(final_scores)) == 1:
                num_filtered_same_score += 1
                continue
            
            filtered_groups[uid] = group
        
        # Post-filter group sizes
        if filtered_groups:
            group_sizes = [len(g) for g in filtered_groups.values()]
            post_filter_metrics = {
                "post_filter/num_groups": len(filtered_groups),
                "post_filter/total_outputs": sum(group_sizes),
                "post_filter/avg_group_size": np.mean(group_sizes),
                "post_filter/min_group_size": np.min(group_sizes),
                "post_filter/max_group_size": np.max(group_sizes),
            }
        else:
            post_filter_metrics = {
                "post_filter/num_groups": 0,
                "post_filter/total_outputs": 0,
                "post_filter/avg_group_size": 0.0,
                "post_filter/min_group_size": 0,
                "post_filter/max_group_size": 0,
            }
        
        total_groups = len(uid_groups)
        filter_metrics = {
            "filter/num_filtered_same_score": num_filtered_same_score,
            "filter/num_filtered_small_size": num_filtered_small_size,
            "filter/total_filtered": num_filtered_same_score + num_filtered_small_size,
            "filter/filter_ratio": (num_filtered_same_score + num_filtered_small_size) / total_groups if total_groups > 0 else 0.0,
            **post_filter_metrics,
        }
        
        return filtered_groups, filter_metrics
    
    def _cap_and_align_outputs(
        self,
        uid_groups: dict[str, list[AgentLoopOutput]],
        target_size: int,
        align_to: int
    ) -> tuple[list[AgentLoopOutput], dict]:
        """Cap total outputs to target_size and align to reranker_n_gpus multiple.
        
        Args:
            uid_groups: Dictionary of uid -> list of outputs.
            target_size: Maximum total outputs (2 * main_batch_size).
            align_to: Must be multiple of this (reranker_n_gpus).
            
        Returns:
            Tuple of (final output list, cap metrics).
        """
        if not uid_groups:
            return [], {"cap/final_total_outputs": 0, "cap/final_num_groups": 0}
        
        total_outputs = sum(len(g) for g in uid_groups.values())
        group_uids = list(uid_groups.keys())
        
        # Case 1: total_outputs > target_size - need to remove groups
        if total_outputs > target_size:
            # Randomly remove groups until we're below target
            random.shuffle(group_uids)
            selected_groups = []
            current_total = 0
            
            for uid in group_uids:
                group = uid_groups[uid]
                if current_total + len(group) <= target_size:
                    selected_groups.append(uid)
                    current_total += len(group)
                else:
                    # Skip this group
                    continue
            
            # Now align to reranker_n_gpus
            final_outputs, num_added = self._align_to_multiple(
                [uid_groups[uid] for uid in selected_groups], align_to
            )
        else:
            # Case 2: total_outputs <= target_size - just align
            final_outputs, num_added = self._align_to_multiple(
                list(uid_groups.values()), align_to
            )
        
        cap_metrics = {
            "cap/original_total_outputs": total_outputs,
            "cap/target_size": target_size,
            "cap/final_total_outputs": len(final_outputs),
            "cap/final_num_groups": len(uid_groups) if total_outputs <= target_size else len(selected_groups),
            "cap/num_outputs_added": num_added,
        }
        
        return final_outputs, cap_metrics
    
    def _align_to_multiple(
        self, 
        groups: list[list[AgentLoopOutput]], 
        align_to: int
    ) -> tuple[list[AgentLoopOutput], int]:
        """Align total outputs to be multiple of align_to by duplicating samples.
        
        Strategy: Balance load across groups when adding duplicates.
        
        Returns:
            Tuple of (final flat output list, num_added).
        """
        # Flatten current outputs
        flat_outputs = []
        for group in groups:
            flat_outputs.extend(group)
        
        current_total = len(flat_outputs)
        remainder = current_total % align_to
        
        if remainder == 0:
            return flat_outputs, 0
        
        # Need to add (align_to - remainder) outputs
        num_to_add = align_to - remainder
        
        # Strategy: randomly select groups and duplicate one sample from each
        # If num_to_add > num_groups, cycle through groups multiple times
        if not groups:
            return flat_outputs, 0
        
        num_groups = len(groups)
        added_outputs = []
        
        for i in range(num_to_add):
            # Cycle through groups if needed
            group_idx = i % num_groups
            group = groups[group_idx]
            
            # Randomly select one output from this group to duplicate
            selected_output = random.choice(group)
            
            # Create a deep copy to avoid reference issues
            import copy
            duplicated_output = copy.deepcopy(selected_output)
            added_outputs.append(duplicated_output)
        
        final_outputs = flat_outputs + added_outputs
        assert len(final_outputs) % align_to == 0, f"Alignment failed: {len(final_outputs)} % {align_to} != 0"
        
        return final_outputs, num_to_add
    
    def _fallback_keep_outputs_to_align(
        self,
        uid_groups: dict[str, list[AgentLoopOutput]],
        align_to: int
    ) -> list[AgentLoopOutput]:
        """Fallback: keep outputs from uid_groups up to align_to multiple.
        
        When all outputs are filtered, use this to ensure we have at least
        align_to outputs for training.
        
        Strategy:
        1. Iterate through uid_groups, append all outputs from each group
        2. When total exceeds align_to, truncate last group to reach exactly align_to
        
        Args:
            uid_groups: Original uid groups before filtering
            align_to: Must be multiple of this (reranker_n_gpus)
            
        Returns:
            List of outputs with length = align_to
        """
        if not uid_groups:
            return []
        
        result_outputs = []
        
        for uid, group in uid_groups.items():
            if len(result_outputs) >= align_to:
                # Already reached target
                break
            
            remaining_slots = align_to - len(result_outputs)
            
            if len(group) <= remaining_slots:
                # Add entire group
                result_outputs.extend(group)
            else:
                # Truncate group to fill exactly to align_to
                result_outputs.extend(group[:remaining_slots])
                break
        
        # Ensure we have exactly align_to outputs
        assert len(result_outputs) == align_to, f"Fallback failed: got {len(result_outputs)}, expected {align_to}"
        
        logger.warning(
            f"Fallback triggered: All outputs were filtered. "
            f"Keeping {len(result_outputs)} outputs from original groups (align_to={align_to})"
        )
        
        return result_outputs
  
    def aggregate_tool_and_agent_score(self, tool_score: float, 
                                        agent_score: float, 
                                        **kwargs) -> float:
        """Aggregate tool score and agent score into final aggregated score.
        
        Args:
            tool_score: Score from tool (reranker).
            agent_score: Score from agent continuation.

        Returns:
            Final aggregated reward score.
        """
        # format penalty (negative tool score indicates format penalty)
        if tool_score < 0:
            return tool_score
        
        # only F1 score > 0.8, we think search-r1 has the correct answer
        binary_agent_score = 1.0 if agent_score >= 0.8 else 0.0
        return max(tool_score, binary_agent_score)
