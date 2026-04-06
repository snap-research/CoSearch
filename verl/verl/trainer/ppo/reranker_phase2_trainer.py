# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Phase 2 Trainer: Train Reranker with fixed Search-R1 as tool.

Alternating training Phase 2:
- Train Reranker using GRPO (single-model, no critic)
- Search-R1 is a fixed tool served via ToolModelManager (standalone vLLM)
- Reward from LLM-as-Judge (or rule-based) scoring the final answer
- Loads training data from Phase 1 trajectories (saved by TrajectorySaver)

Key differences from Phase 1:
1. Trains Reranker instead of Search-R1
2. Data comes from saved trajectories (RerankerTrainingDataset), not a standard RL dataset
3. Each sample = (initial_query, sub_query, top-50 docs) from a Phase 1 tool call
4. Simple UID grouping: hash(initial_query + sub_query + step_index) for GRPO
5. Optional: Search-R1 continuation for end-to-end reward
"""

import asyncio
import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional, Any

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from torch.utils.data import Dataset, Sampler, DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

# Shared helpers from the joint trainer
from verl.trainer.ppo.search_r1_reranker_reward_ray_trainer import (
    apply_kl_penalty,
    compute_response_mask,
    compute_advantage,
)


@dataclass
class Phase2ResourcePoolManager:
    """Resource pool specification for Phase 2 (Reranker only training).
    
    Only manages the reranker training pool.
    Search-R1 and LLM-as-Judge run via ToolModelManager/LLMJudgeRewardModelManager (standalone).
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=2, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


class RerankerPhase2Trainer(RayPPOTrainer):
    """Phase 2 Trainer: Train Reranker with fixed Search-R1 as tool.

    Single-model GRPO training for the Reranker. Search-R1 is served as a fixed
    tool via ToolModelManager (standalone vLLM servers on dedicated GPUs).
    
    Training data comes from Phase 1 trajectories loaded by RerankerTrainingDataset.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: Phase2ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        device_name=None,
    ):
        """Initialize Phase 2 trainer.

        Args:
            config: Configuration object.
            tokenizer: Tokenizer for the Reranker model.
            role_worker_mapping: Mapping from roles to worker classes (ActorRollout for reranker).
            resource_pool_manager: Manager for reranker resource pool.
            ray_worker_group_cls: Class for Ray worker groups.
            processor: Optional data processor.
            train_dataset: Training dataset (RerankerTrainingDataset).
            val_dataset: Validation dataset (optional).
            collate_fn: Collation function.
            device_name: Device name.
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_critic = False  # Phase 2 doesn't use critic

        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # Reference policy: if using LoRA, ref is actor without LoRA applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # KL control
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)
        else:
            self.kl_ctrl_in_reward = None

        self._create_dataloader(train_dataset, val_dataset, collate_fn)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn):
        """Create dataloaders from Phase 1 trajectory data."""
        if train_dataset is None:
            # Load from trajectory store
            from verl.experimental.trajectory_store.trajectory_dataset import RerankerTrainingDataset

            trajectory_dir = self.config.data.get("trajectory_dir")
            assert trajectory_dir, "trajectory_dir must be set for Phase 2 training"

            train_dataset = RerankerTrainingDataset(
                trajectory_dir=trajectory_dir,
                step_range=self.config.data.get("trajectory_step_range", None),
                min_documents=self.config.data.get("min_documents", 5),
                top_k_sub_queries=self.config.data.get("top_k_sub_queries", 4),
                total_samples=self.config.data.get("total_samples", None),
                reranker_top_m=self.config.get("reranker_top_m", 5),
                seed=self.config.data.get("dataset_seed", 42),
            )
            print(f"[Phase2] Loaded {len(train_dataset)} samples from trajectories at {trajectory_dir}")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if collate_fn is None:
            from verl.experimental.trajectory_store.trajectory_dataset import RerankerTrainingDataset
            collate_fn = RerankerTrainingDataset.collate_fn

        num_workers = self.config.data.get("dataloader_num_workers", 0)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            shuffle=True,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        print(f"Size of train dataloader: {len(self.train_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config: {e}")

    def init_workers(self):
        """Initialize distributed training workers for Phase 2.

        Architecture (GPU placement strategy):
          1. Reranker PG [16] STRICT_PACK created FIRST → claims one entire node
          2. ToolModelManager (Search-R1, pure vLLM) → only the other node has free GPUs
          3. LLMJudgeRewardModelManager (pure vLLM) → same node as tool
          4. RayWorkerGroup spawns reranker workers using the PG from step 1

        This ordering guarantees node isolation: reranker gets a whole node,
        tool + judge share the other node. No node affinity API needed.
        """
        # =====================================================================
        # Step 1: Create reranker PG FIRST — claim one entire node (16 GPUs)
        #
        # RayResourcePool.get_placement_groups() is lazy (only creates PG on
        # first call). We call it eagerly here so that the [16] STRICT_PACK PG
        # is scheduled while the cluster is still fully idle (32 GPUs free).
        # This guarantees the PG lands on one node. Tool/judge created later
        # can only go to the other node.
        # =====================================================================
        print("[Phase2] Creating reranker resource pool FIRST (before tools)...")
        self.resource_pool_manager.create_resource_pool()
        for pool_name, pool in self.resource_pool_manager.resource_pool_dict.items():
            print(f"[Phase2] Eagerly creating PG for pool '{pool_name}' "
                  f"({pool.world_size} GPUs, STRICT_PACK)...")
            pool.get_placement_groups()  # blocks until PG is ready
            print(f"[Phase2] ✓ Pool '{pool_name}' PG created and ready")

        # =====================================================================
        # Step 2: Create ToolModelManager for Search-R1 (standalone pure vLLM)
        # Now only the other node has free GPUs, so all tool PGs land there.
        # =====================================================================
        search_r1_tool_config = self.config.get("search_r1_tool_model", None)
        if search_r1_tool_config is not None and search_r1_tool_config.get("enable", True):
            from verl.experimental.tool_model.tool_model_manager import ToolModelManager

            print("[Phase2] Creating ToolModelManager for Search-R1 (standalone vLLM)...")
            print(f"[Phase2] Search-R1 model: {search_r1_tool_config.model.path}")

            self.search_r1_tool_manager = ToolModelManager(
                config=search_r1_tool_config,
                worker_group=None,
            )
            print(f"[Phase2] ✓ Search-R1 ToolModelManager claimed GPUs with "
                  f"{len(self.search_r1_tool_manager.server_handles)} handles")
        else:
            self.search_r1_tool_manager = None
            print("[Phase2] No Search-R1 tool model configured (reranker-only reward)")

        # =====================================================================
        # Step 3: Create LLMJudgeRewardModelManager (standalone pure vLLM)
        # Same node as tool — the reranker node is fully occupied.
        # =====================================================================
        self.reward_model_manager = None
        reward_judge_config = self.config.get("reward_judge_model", None)
        if reward_judge_config is not None and reward_judge_config.get("enable", False):
            reward_judge_mode = reward_judge_config.get("mode", "http_server")

            if reward_judge_mode == "http_server":
                server_urls = reward_judge_config.get("server_urls", ["http://localhost:8000"])
                if isinstance(server_urls, str):
                    server_urls = [server_urls]
                print(f"[Phase2] Reward judge: HTTP mode ({server_urls})")
                self.reward_http_server_urls = server_urls
            else:
                from verl.experimental.reward.llm_judege_reward_model import LLMJudgeRewardModelManager
                print(f"[Phase2] Creating LLMJudgeRewardModelManager (standalone)...")
                self.reward_model_manager = LLMJudgeRewardModelManager(
                    config=reward_judge_config,
                    worker_group=None,
                )
                print(f"[Phase2] ✓ Reward model claimed GPUs")
                self.reward_http_server_urls = None
        else:
            self.reward_http_server_urls = None
            print("[Phase2] No reward judge configured (using rule-based reward)")

        # =====================================================================
        # Step 4: Register role→class mappings (PGs already created in Step 1)
        # =====================================================================
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Reranker actor+rollout
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role=str(Role.ActorRollout),
        )
        self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls

        # Reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # =====================================================================
        # Step 4: Initialize worker groups
        # =====================================================================
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                )
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            print("=" * 80)
            print(f"[Phase2] Resource pool: {resource_pool.name_prefix if hasattr(resource_pool, 'name_prefix') else resource_pool}")
            print(f"[Phase2] Pool size: {resource_pool.world_size} GPUs")
            print(f"[Phase2] Roles: {list(class_dict.keys())}")
            print("=" * 80)

            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        # Initialize reference policy
        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()

        # Initialize reranker (last for KV cache estimation)
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # =====================================================================
        # Step 5: Create Phase 2 agent loop manager
        # =====================================================================
        assert self.config.actor_rollout_ref.rollout.mode == "async", \
            "Phase 2 trainer only supports async rollout mode"

        self.async_rollout_mode = True
        self.async_rollout_manager = Phase2AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            search_r1_tool_manager=self.search_r1_tool_manager,
            reward_model_manager=self.reward_model_manager,
            reward_http_server_urls=getattr(self, 'reward_http_server_urls', None),
        )

    def _save_checkpoint(self):
        """Save Reranker checkpoint."""
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        print(f"local_global_step_folder: {local_global_step_folder}")

        actor_local_path = os.path.join(local_global_step_folder, "reranker_actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "reranker_actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        # Save dataloader state
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        """Load checkpoint for resuming training."""
        if self.config.trainer.resume_mode == "disable":
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs not implemented")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str)
            assert "global_step_" in self.config.trainer.resume_from_path
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)

        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Setting global step to {self.global_steps}")

        actor_path = os.path.join(global_step_folder, "reranker_actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )

        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder data for balanced DP."""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)
        workload_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            workload_lst, k_partitions=world_size, equal_size=True
        )
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Override for Phase 2: use answers as ground truth, add judge details.

        Base class uses reward_model.ground_truth for gts, which Phase 2
        doesn't have.  Phase 2 stores answers in non_tensor_batch['answers'].
        Also adds uid, reward_source, answer_in_docs, judge_prompt, and
        judge_response to the JSONL dump.
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

            # Phase 2: use 'answers' as ground truth (not reward_model.ground_truth)
            sample_gts = list(batch.non_tensor_batch.get("answers", [None] * len(inputs)))

            # Build extra info dict with Phase 2-specific fields
            extra_infos = reward_extra_infos_dict.copy()
            for key in ["uid", "reward_source", "answer_in_docs", "judge_prompt", "judge_response"]:
                if key in batch.non_tensor_batch:
                    extra_infos[key] = batch.non_tensor_batch[key]

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=extra_infos,
                dump_path=rollout_data_dir,
            )

    def fit(self):
        """Main training loop for Phase 2.
        
        Single-model GRPO training for Reranker with:
        - Training data from Phase 1 trajectories
        - N rollouts per sample for GRPO grouping
        - Optional Search-R1 continuation + LLM-as-Judge reward
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Phase 2 Training")

        self.global_steps += 1
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                # batch_dict comes from RerankerTrainingDataset collate_fn
                # It contains: initial_query, sub_query, top_50_documents, answers etc.
                # All values are plain Python lists (strings, dicts, floats), NOT tensors.
                # We must convert to DataProto manually (not via from_single_dict which expects tensors).

                if isinstance(batch_dict, dict):
                    # Convert all list fields to numpy 1D object arrays for non_tensor_batch.
                    # IMPORTANT: Do NOT use np.array(values, dtype=object) directly!
                    # When all elements have the same length (e.g. every sample has
                    # exactly 50 documents), numpy auto-promotes to a 2D array and
                    # v[i] returns an ndarray instead of the original Python object.
                    # The np.empty + slice-assign idiom forces a 1D object array.
                    non_tensor_batch = {}
                    for key, values in batch_dict.items():
                        arr = np.empty(len(values), dtype=object)
                        arr[:] = values
                        non_tensor_batch[key] = arr
                    batch_size = len(next(iter(batch_dict.values())))
                    # Create a minimal DataProto with only non_tensor_batch
                    # (no tensor batch — Phase 2 worker will create prompts and generate responses)
                    batch = DataProto(
                        batch=TensorDict({}, batch_size=[batch_size]),
                        non_tensor_batch=non_tensor_batch,
                        meta_info={},
                    )
                else:
                    batch = batch_dict

                # Ensure UIDs exist
                assert "uid" not in batch.non_tensor_batch, "UID should not be pre-generated in the dataset for Phase 2, trainer will generate its own UIDs for GRPO grouping"
                if "uid" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.non_tensor_batch[next(iter(batch.non_tensor_batch))]))], dtype=object
                    )
                gen_batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # Generate reranker rollouts
                    with marked_timer("gen", timing_raw, color="red"):
                        reranker_batch = self.async_rollout_manager.generate_sequences(gen_batch)

                        if "timing" in reranker_batch.meta_info:
                            timing_raw.update(reranker_batch.meta_info.pop("timing"))
                        if "aggregated_metrics" in reranker_batch.meta_info:
                            gen_metrics = reranker_batch.meta_info.pop("aggregated_metrics")
                            metrics.update({f"reranker_{k}": v for k, v in gen_metrics.items()})
                        # Extract filtering stats for wandb logging later
                        filtering_stats = reranker_batch.meta_info.pop("filtering_stats", {})

                    if "response_mask" not in reranker_batch.batch.keys():
                        reranker_batch.batch["response_mask"] = compute_response_mask(reranker_batch)

                    # Balance tokens
                    if self.config.trainer.balance_batch:
                        self._balance_batch(reranker_batch, metrics=metrics)

                    # ============================================================
                    # Single-agent GRPO step for reranker
                    # ============================================================
                    reranker_batch.meta_info["global_token_num"] = torch.sum(
                        reranker_batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # Extract reward
                    with marked_timer("reward", timing_raw, color="yellow"):
                        assert "rm_scores" in reranker_batch.batch.keys(), \
                            "rm_scores should be set by Phase 2 agent loop worker"
                        reward_tensor = reranker_batch.batch["rm_scores"]
                        reward_extra_keys = reranker_batch.meta_info.get("reward_extra_keys", [])
                        reward_extra_infos_dict = (
                            {key: reranker_batch.non_tensor_batch[key] for key in reward_extra_keys}
                            if reward_extra_keys else {}
                        )

                    # Compute old_log_probs
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = (
                        rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    )

                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction
                        apply_rollout_correction(
                            batch=reranker_batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(reranker_batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = reranker_batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            metrics["actor/entropy"] = entropy_agg.detach().item()
                            old_log_prob.batch.pop("entropys")
                            reranker_batch = reranker_batch.union(old_log_prob)
                            if "rollout_log_probs" in reranker_batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics
                                metrics.update(calculate_debug_metrics(reranker_batch))

                    assert "old_log_probs" in reranker_batch.batch

                    # Reference log_prob
                    if self.use_reference_policy:
                        with marked_timer("ref_log_prob", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(reranker_batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(reranker_batch)
                            reranker_batch = reranker_batch.union(ref_log_prob)

                    # Compute advantages (GRPO)
                    with marked_timer("adv", timing_raw, color="brown"):
                        reranker_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            reranker_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        if self.config.algorithm.use_kl_in_reward:
                            reranker_batch, kl_metrics = apply_kl_penalty(
                                reranker_batch, kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            reranker_batch.batch["token_level_rewards"] = reranker_batch.batch["token_level_scores"]

                        # Rollout correction: IS weights, rejection sampling (decoupled mode)
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in reranker_batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
                            reranker_batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                reranker_batch, rollout_corr_config
                            )
                            metrics.update(is_metrics)

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        reranker_batch = compute_advantage(
                            reranker_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # Update actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            reranker_batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            reranker_batch.meta_info["temperature"] = rollout_config.temperature
                            actor_output = self.actor_rollout_wg.update_actor(reranker_batch)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update({k: v for k, v in actor_output_metrics.items()})

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(reranker_batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    # Rollout stats
                    if "token_level_scores" in reranker_batch.batch:
                        scores = reranker_batch.batch["token_level_scores"].sum(-1).cpu().numpy()
                        metrics["reranker/score_mean"] = float(np.mean(scores))
                    if "rerank_success" in reward_extra_infos_dict:
                        success_arr = np.array(reward_extra_infos_dict["rerank_success"])
                        metrics["reranker/rerank_success_rate"] = float(np.mean(success_arr))
                    if "answer_in_reranked" in reward_extra_infos_dict:
                        aid_arr = np.array(reward_extra_infos_dict["answer_in_reranked"])
                        metrics["reranker/answer_in_reranked_rate"] = float(np.mean(aid_arr))

                    # Filtering & judge stats from agent loop workers
                    if filtering_stats:
                        metrics["reranker/n_total_rollouts"] = filtering_stats.get("n_total", 0)
                        metrics["reranker/n_dropped_judge_fail"] = filtering_stats.get("n_dropped_judge_fail", 0)
                        metrics["reranker/n_dropped_group_prune"] = filtering_stats.get("n_dropped_group_prune", 0)
                        metrics["reranker/n_valid_rollouts"] = filtering_stats.get("n_valid", 0)
                        metrics["reranker/judge_calls_total"] = filtering_stats.get("judge_calls_total", 0)
                        metrics["reranker/judge_calls_success"] = filtering_stats.get("judge_calls_success", 0)
                        jt = filtering_stats.get("judge_calls_total", 0)
                        if jt > 0:
                            metrics["reranker/judge_success_rate"] = (
                                filtering_stats["judge_calls_success"] / jt
                            )

                    # Reward breakdown by source
                    if "reward_source" in reranker_batch.non_tensor_batch:
                        reward_sources = reranker_batch.non_tensor_batch["reward_source"]
                        scores_np = reranker_batch.batch["token_level_scores"].sum(-1).cpu().numpy()
                        for source in ["format_penalty", "answer_in_docs", "judge", "no_judge_backend"]:
                            mask = np.array([str(s) == source for s in reward_sources])
                            if mask.any():
                                metrics[f"reranker/reward_{source}_count"] = int(mask.sum())
                                metrics[f"reranker/reward_{source}_mean"] = float(scores_np[mask].mean())

                # Save checkpoint
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=reranker_batch, use_critic=False, agent_name="reranker"))
                metrics.update(compute_timing_metrics(batch=reranker_batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=reranker_batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    progress_bar.close()
                    return


class Phase2AgentLoopManager:
    """Agent loop manager for Phase 2 reranker training.
    
    Manages reranker vLLM servers (hybrid with training worker group) and
    optional Search-R1 tool servers + reward judge servers (standalone).
    """

    def __init__(
        self,
        config,
        worker_group: RayWorkerGroup,
        search_r1_tool_manager=None,
        reward_model_manager=None,
        reward_http_server_urls=None,
    ):
        from verl.experimental.agent_loop.reranker_phase2_agent_loop_worker import RerankerPhase2AgentLoopWorker
        from verl.workers.rollout.replica import get_rollout_replica_class

        self.config = config
        self.worker_group = worker_group
        self.search_r1_tool_manager = search_r1_tool_manager
        self.reward_model_manager = reward_model_manager
        self.reward_http_server_urls = reward_http_server_urls

        # Initialize reranker vLLM servers (hybrid with FSDP)
        self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        self._initialize_reranker_servers()

        # Get search-R1 tool server handles
        self.search_r1_server_handles = (
            self.search_r1_tool_manager.server_handles if self.search_r1_tool_manager else None
        )

        # Get reward server handles
        self.reward_server_handles = (
            self.reward_model_manager.server_handles if self.reward_model_manager else None
        )

        # Initialize workers
        self._init_workers()

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _initialize_reranker_servers(self):
        """Initialize reranker vLLM/SGLang servers (hybrid with training worker group)."""
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = self.worker_group.world_size
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model

        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        import asyncio
        loop = asyncio.new_event_loop()

        async def _init_all():
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])

        loop.run_until_complete(_init_all())
        loop.close()

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"[Phase2Manager] Reranker servers at {self.server_addresses}")

    def _init_workers(self):
        """Initialize Phase 2 agent loop workers."""
        from verl.experimental.agent_loop.reranker_phase2_agent_loop_worker import RerankerPhase2AgentLoopWorker

        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [
            node["NodeID"] for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("CPU", 0) > 0
        ]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                RerankerPhase2AgentLoopWorker.options(
                    name=f"phase2_reranker_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    config=self.config,
                    reranker_server_handles=self.server_handles,
                    search_r1_server_handles=self.search_r1_server_handles,
                    reward_server_handles=self.reward_server_handles,
                    reward_http_server_urls=self.reward_http_server_urls,
                )
            )

    def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate reranker rollouts from Phase 2 samples."""
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()

        chunks = batch.chunk(len(self.agent_loop_workers))
        outputs = ray.get([
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks, strict=True)
        ])

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # Aggregate metrics across all workers, popping per-worker keys
        # BEFORE DataProto.concat (which asserts meta_info values are identical).
        timing = {}
        aggregated_metrics = {}
        agg_filtering_stats = {}
        for output in outputs:
            if "timing" in output.meta_info:
                for k, v in output.meta_info.pop("timing").items():
                    timing.setdefault(k, [])
                    timing[k].append(v)
            # Sum filtering stats across workers
            if "filtering_stats" in output.meta_info:
                for k, v in output.meta_info.pop("filtering_stats").items():
                    agg_filtering_stats[k] = agg_filtering_stats.get(k, 0) + v
        timing = {k: sum(v) / len(v) for k, v in timing.items()}

        # Filter out empty DataProtos (workers where all rollouts were filtered)
        non_empty_outputs = [o for o in outputs if len(o) > 0]
        if not non_empty_outputs:
            print("[Phase2Manager] WARNING: All workers returned empty batches after filtering!")
            result = DataProto()
            result.meta_info = {
                "timing": timing,
                "aggregated_metrics": aggregated_metrics,
                "filtering_stats": agg_filtering_stats,
            }
            return result

        result = DataProto.concat(non_empty_outputs)

        # Align batch size to world_size (FSDP requires equal partitions).
        # After filtering, the total may no longer be divisible by world_size.
        # Duplicate random samples to pad up to the next multiple.
        world_size = self.worker_group.world_size
        n = len(result)
        remainder = n % world_size
        if remainder != 0:
            num_to_add = world_size - remainder
            # Pick random indices to duplicate (with replacement if needed)
            pad_indices = np.random.choice(n, size=num_to_add, replace=(num_to_add > n)).tolist()
            pad_batch = deepcopy(result[pad_indices])
            result = DataProto.concat([result, pad_batch])
            agg_filtering_stats["n_padded_for_alignment"] = num_to_add
            print(
                f"[Phase2Manager] Padded {num_to_add} samples for world_size={world_size} "
                f"alignment: {n} → {len(result)}"
            )

        result.meta_info["timing"] = timing
        result.meta_info["aggregated_metrics"] = aggregated_metrics
        result.meta_info["filtering_stats"] = agg_filtering_stats

        return result

    def wake_up(self):
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])
        if self.search_r1_tool_manager:
            self.search_r1_tool_manager.wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()

    def sleep(self):
        self._run_all([replica.sleep() for replica in self.rollout_replicas])
        if self.search_r1_tool_manager:
            self.search_r1_tool_manager.sleep()
        if self.reward_model_manager:
            self.reward_model_manager.sleep()

    def _run_all(self, tasks: list):
        """Run a list of async tasks synchronously."""
        async def _gather():
            return await asyncio.gather(*tasks)
        asyncio.run(_gather())
