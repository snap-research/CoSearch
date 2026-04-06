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
Phase 1 Trainer: Train Search-R1 with fixed reranker as tool.

Alternating training Phase 1:
- Train Search-R1 using GRPO (single-model, no critic)
- Reranker is a fixed tool served via ToolModelManager (standalone vLLM)
- No reward model (LLM-as-Judge) — reward is rule-based (e.g. F1 from agent loop)
- Trajectory data (including top-50 documents) is saved for Phase 2 reranker training

Key differences from SearchR1RerankerRewardRayTrainer:
1. Single-model training (Search-R1 only), no dual-agent PPO
2. Reranker served via ToolModelManager (standalone vLLM, own GPU pool)
3. TrajectorySaver hook after generate_sequences
4. Simpler init_workers: only main agent resource pool + ToolModelManager
5. No reward judge model (rule-based reward only)
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
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

# Import from the joint trainer for shared helper functions
from verl.trainer.ppo.search_r1_reranker_reward_ray_trainer import (
    ResourcePoolManager,
    apply_kl_penalty,
    compute_response_mask,
    compute_advantage,
)


@dataclass
class Phase1ResourcePoolManager:
    """Resource pool specification for Phase 1 (Search-R1 only training).
    
    Only manages the main agent (Search-R1) pool.
    The reranker runs via ToolModelManager (standalone, self-managed GPUs).
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


class SearchR1Phase1Trainer(RayPPOTrainer):
    """Phase 1 Trainer: Train Search-R1 with fixed reranker as tool.

    Single-model GRPO training for Search-R1. The reranker is served as a fixed
    tool via ToolModelManager (standalone vLLM servers on dedicated GPUs).
    
    Trajectories (including top-50 documents per tool call) are saved to disk
    for Phase 2 reranker training.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: Phase1ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """Initialize Phase 1 trainer.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer for Search-R1.
            role_worker_mapping: Mapping from roles to worker classes (only ActorRollout needed).
            resource_pool_manager: Manager for Ray resource pools (main agent only).
            ray_worker_group_cls: Class for Ray worker groups.
            processor: Optional data processor for multimodal data.
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            collate_fn: Function to collate data samples into batches.
            train_sampler: Sampler for the training dataset.
            device_name: Device name for training.
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        assert not self.use_critic, "Phase 1 trainer doesn't support critic."

        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

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

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # Phase 1 assertions
        assert not self.use_rm, "Phase 1 trainer uses rule-based reward from agent loop, not standalone reward model."

        # TrajectorySaver will be initialized after init_workers (needs reranker_tokenizer from ToolModelManager)
        self.trajectory_saver = None

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """Creates the train and validation dataloaders."""
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, "
            f"Size of val dataloader: {len(self.val_dataloader)}"
        )

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

    def _make_json_serializable(self, obj):
        """Convert numpy/torch types to native Python types for JSON serialization."""
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (torch.Tensor,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if k in base_data:
                continue
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: self._make_json_serializable(v[i]) for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Dumped generations to {filename}")

    def _log_rollout_data(self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str):
        """Log rollout data to disk."""
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )
            if "uid" in batch.non_tensor_batch:
                reward_extra_infos_to_dump["uid"] = batch.non_tensor_batch["uid"]

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _compute_rollout_stats(self, batch: DataProto, reward_extra_infos_dict: dict, prefix: str = "") -> dict:
        """Compute statistics from rollout data for tracking."""
        stats = {}
        if "valid" in reward_extra_infos_dict:
            valid_array = np.array(reward_extra_infos_dict["valid"])
            valid_count = np.sum(valid_array == 1)
            total_count = len(valid_array)
            valid_rate = valid_count / total_count if total_count > 0 else 0.0
            stats[f"{prefix}/valid_rate"] = valid_rate
        if "token_level_scores" in batch.batch:
            scores = batch.batch["token_level_scores"].sum(-1).cpu().numpy()
            stats[f"{prefix}/score_mean"] = float(np.mean(scores))
        if "f1" in reward_extra_infos_dict:
            f1_array = np.array(reward_extra_infos_dict["f1"])
            stats[f"{prefix}/f1_mean"] = float(np.mean(f1_array))
        if "agent_reward" in reward_extra_infos_dict:
            agent_reward_array = np.array(reward_extra_infos_dict["agent_reward"])
            stats[f"{prefix}/agent_reward_mean"] = float(np.mean(agent_reward_array))
        if "tool_reward" in reward_extra_infos_dict:
            tool_reward_array = np.array(reward_extra_infos_dict["tool_reward"])
            stats[f"{prefix}/tool_reward_mean"] = float(np.mean(tool_reward_array))
        return stats

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger."""
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[:generations_to_log]
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(self, batch: DataProto, reward_fn=None, return_dict: bool = False, sum_reward: bool = False):
        """Compute or extract reward from batch."""
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            if return_dict:
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                return reward_tensor, reward_extra_infos_dict

        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")
        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """Extract generation batch from full batch."""
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        return gen_batch

    def _validate(self):
        """Run validation."""
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs, sample_outputs, sample_gts, sample_scores, sample_turns, sample_uids = [], [], [], [], [], []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs, outputs=sample_outputs, gts=sample_gts,
                scores=sample_scores, reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers.

        Architecture for Phase 1 (GPU placement strategy):
          1. Training PG [16] STRICT_PACK created FIRST → claims one entire node
          2. (Optional) ToolModelManager (reranker, standalone vLLM) → only the other node has free GPUs
          3. Phase1AgentLoopManager connects main agent servers + optional reranker tool servers

        This ordering guarantees node isolation: training gets a whole node,
        reranker tool lands on the other node. Same pattern as Phase 2.

        If reranker_tool_model is not configured or enable=False, runs in
        retrieval-only mode (no reranker tool, same as train_search_r1_grpo.sh).
        """
        # =====================================================================
        # Step 1: Create training PG FIRST — claim one entire node (16 GPUs)
        #
        # RayResourcePool.get_placement_groups() is lazy (only creates PG on
        # first call). We call it eagerly here so that the [16] STRICT_PACK PG
        # is scheduled while the cluster is still fully idle (32 GPUs free).
        # This guarantees the PG lands on one node. Tool PGs created later
        # can only go to the other node.
        # =====================================================================
        reranker_tool_config = self.config.get("reranker_tool_model", None)
        use_reranker = (
            reranker_tool_config is not None
            and reranker_tool_config.get("enable", True)
        )

        print("[Phase1] Creating training resource pool FIRST (before tools)...")
        self.resource_pool_manager.create_resource_pool()
        for pool_name, pool in self.resource_pool_manager.resource_pool_dict.items():
            print(f"[Phase1] Eagerly creating PG for pool '{pool_name}' "
                  f"({pool.world_size} GPUs, STRICT_PACK)...")
            pool.get_placement_groups()  # blocks until PG is ready
            print(f"[Phase1] ✓ Pool '{pool_name}' PG created and ready")

        # =====================================================================
        # Step 2: Optionally create ToolModelManager for reranker (standalone vLLM)
        # Now only the other node has free GPUs, so all tool PGs land there.
        # =====================================================================
        self.tool_model_manager = None
        self.reranker_model_path = None

        if use_reranker:
            from verl.experimental.tool_model.tool_model_manager import ToolModelManager

            print("[Phase1] Creating ToolModelManager for reranker (standalone vLLM)...")
            print(f"[Phase1] Reranker model: {reranker_tool_config.model.path}")
            print(f"[Phase1] TP size: {reranker_tool_config.rollout.tensor_model_parallel_size}")
            print(f"[Phase1] GPUs: {reranker_tool_config.n_gpus_per_node} x {reranker_tool_config.nnodes} node(s)")

            self.tool_model_manager = ToolModelManager(
                config=reranker_tool_config,
                worker_group=None,  # standalone mode — creates own RayResourcePool
            )
            self.reranker_model_path = reranker_tool_config.model.path
            print(f"[Phase1] ✓ ToolModelManager claimed GPUs with "
                  f"{len(self.tool_model_manager.server_handles)} server handles")
        else:
            print("[Phase1] No reranker tool configured — retrieval-only mode")

        # =====================================================================
        # Step 3: Register role→class mappings (PGs already created in Step 1)
        # =====================================================================
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Create main agent actor+rollout
        assert self.hybrid_engine
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role=str(Role.ActorRollout),
        )
        self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls

        # Create reference policy if needed
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
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            print("=" * 80)
            print(f"[Phase1] Resource pool: {resource_pool.name_prefix if hasattr(resource_pool, 'name_prefix') else resource_pool}")
            print(f"[Phase1] Pool size: {resource_pool.world_size} GPUs")
            print(f"[Phase1] Roles: {list(class_dict.keys())}")
            print("=" * 80)

            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            print(f"[Phase1] ✓ Created {len(spawn_wg)} worker groups")

        # Initialize reference policy
        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()

        self.rm_wg = None

        # Initialize main agent (last, so vLLM can estimate KV cache memory)
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # =====================================================================
        # Step 5: Create async rollout manager with optional reranker tool
        # =====================================================================
        assert self.config.actor_rollout_ref.rollout.mode == "async", \
            "Phase 1 trainer only supports async rollout mode"

        self.async_rollout_mode = True

        self.async_rollout_manager = Phase1AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            tool_model_manager=self.tool_model_manager,  # None if retrieval-only
            reranker_model_path=self.reranker_model_path,  # None if retrieval-only
            rm_wg=self.rm_wg,
        )

        # =====================================================================
        # Step 6: Initialize TrajectorySaver
        # =====================================================================
        trajectory_config = self.config.get("trajectory", None)
        trajectory_dir = trajectory_config.get("save_dir", None) if trajectory_config else None
        if trajectory_dir:
            from verl.experimental.trajectory_store.trajectory_saver import TrajectorySaver
            self.trajectory_saver = TrajectorySaver(
                output_dir=trajectory_dir,
                compress=trajectory_config.get("compress", True),
            )
            print(f"[Phase1] TrajectorySaver initialized: {trajectory_dir}")
        else:
            print("[Phase1] WARNING: trajectory.save_dir not set. No trajectories will be saved for Phase 2.")

    def _save_checkpoint(self):
        """Save Search-R1 checkpoint (main agent only)."""
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        print(f"local_global_step_folder: {local_global_step_folder}")

        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
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

        # Write latest checkpointed iteration
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
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str)
                assert "global_step_" in self.config.trainer.resume_from_path
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)

        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Setting global step to {self.global_steps}")

        actor_path = os.path.join(global_step_folder, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )

        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}")

    def _save_trajectories(self, batch: DataProto):
        """Save trajectory data for Phase 2 reranker training.
        
        Extracts tool_call_details, messages, answers from the batch's extra fields
        and saves them as JSONL via TrajectorySaver.
        """
        if self.trajectory_saver is None:
            return

        from verl.experimental.trajectory_store.trajectory_saver import TrajectorySaver

        trajectories = []
        batch_size = batch.batch.batch_size[0]
        
        for i in range(batch_size):
            item = batch[i]
            
            # Extract extra fields that were packed by the agent loop
            extra_fields = {}
            for key in ["tool_call_details", "messages", "initial_query", "answers"]:
                if key in item.non_tensor_batch:
                    extra_fields[key] = item.non_tensor_batch[key]

            # Only save trajectories that have tool_call_details (i.e. save_top_n_documents was True)
            if not extra_fields.get("tool_call_details"):
                continue

            # Compute final reward (sum of token-level scores)
            final_reward = float(item.batch["token_level_scores"].sum().item()) if "token_level_scores" in item.batch else 0.0

            uid = item.non_tensor_batch.get("uid", f"unknown_{i}")

            # Compute num_turns from tool_call_details or messages
            tool_call_details = extra_fields["tool_call_details"]
            num_turns = len(tool_call_details)

            trajectory = TrajectorySaver.build_trajectory_from_rollout(
                step=self.global_steps,
                uid=str(uid),
                initial_query=extra_fields.get("initial_query", ""),
                answers=extra_fields.get("answers", []),
                messages=extra_fields.get("messages", []),
                tool_call_details=tool_call_details,
                final_reward=final_reward,
                num_turns=num_turns,
            )
            trajectories.append(trajectory)

        if trajectories:
            self.trajectory_saver.save_step(self.global_steps, trajectories)
            print(f"[Phase1] Saved {len(trajectories)} trajectories for step {self.global_steps}")
        else:
            print(f"[Phase1] No trajectories with tool_call_details for step {self.global_steps}")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder data so each dp rank gets similar total tokens."""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)
        workload_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size: (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
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

    def _start_profiling(self, do_profile: bool) -> None:
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()

    def fit(self):
        """Main training loop for Phase 1.
        
        Single-model GRPO training for Search-R1 with:
        - Fixed reranker served as tool via ToolModelManager
        - Rule-based reward (F1 from agent loop)
        - Trajectory saving for Phase 2
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

        # Validate before training
        if (self.val_reward_fn or self.use_reward_loop) and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Phase 1 Training")

        self.global_steps += 1
        last_val_metrics = None
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

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # Generate rollouts
                    with marked_timer("gen", timing_raw, color="red"):
                        assert self.async_rollout_mode
                        # Phase 1: only main agent rollout (reranker is tool, no reranker batch)
                        main_batch = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update({k: v for k, v in main_batch.meta_info["timing"].items()})
                        main_batch.meta_info.pop("timing", None)

                        if "aggregated_metrics" in main_batch.meta_info:
                            main_metrics = main_batch.meta_info.pop("aggregated_metrics")
                            metrics.update({k: v for k, v in main_metrics.items()})

                    # Save trajectories for Phase 2 (before any modifications to batch)
                    with marked_timer("save_trajectories", timing_raw, color="cyan"):
                        self._save_trajectories(main_batch)

                    if "response_mask" not in main_batch.batch.keys():
                        main_batch.batch["response_mask"] = compute_response_mask(main_batch)

                    # Balance tokens across DP ranks
                    if self.config.trainer.balance_batch:
                        self._balance_batch(main_batch, metrics=metrics)

                    # ============================================================
                    # Single-agent GRPO step (inlined from process_single_agent_ppo_step)
                    # ============================================================
                    main_batch.meta_info["global_token_num"] = torch.sum(
                        main_batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # Extract reward (already computed in agent loop)
                    with marked_timer("reward", timing_raw, color="yellow"):
                        assert "rm_scores" in main_batch.batch.keys()
                        reward_tensor = main_batch.batch["rm_scores"]
                        reward_extra_keys = main_batch.meta_info.get("reward_extra_keys", [])
                        reward_extra_infos_dict = (
                            {key: main_batch.non_tensor_batch[key] for key in reward_extra_keys}
                            if reward_extra_keys else {}
                        )

                    # Compute old_log_probs
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)

                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction
                        apply_rollout_correction(
                            batch=main_batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(main_batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = main_batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            metrics["actor/entropy"] = entropy_agg.detach().item()
                            old_log_prob.batch.pop("entropys")
                            main_batch = main_batch.union(old_log_prob)
                            if "rollout_log_probs" in main_batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics
                                metrics.update(calculate_debug_metrics(main_batch))

                    assert "old_log_probs" in main_batch.batch

                    # Compute reference log_prob if needed
                    if self.use_reference_policy:
                        with marked_timer("ref_log_prob", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(main_batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(main_batch)
                            main_batch = main_batch.union(ref_log_prob)

                    # Compute advantages (GRPO)
                    with marked_timer("adv", timing_raw, color="brown"):
                        main_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            main_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        if self.config.algorithm.use_kl_in_reward:
                            main_batch, kl_metrics = apply_kl_penalty(
                                main_batch, kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            main_batch.batch["token_level_rewards"] = main_batch.batch["token_level_scores"]

                        # Rollout correction
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in main_batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
                            main_batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                main_batch, rollout_corr_config
                            )
                            metrics.update(is_metrics)

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        main_batch = compute_advantage(
                            main_batch,
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
                            main_batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            main_batch.meta_info["temperature"] = rollout_config.temperature
                            actor_output = self.actor_rollout_wg.update_actor(main_batch)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Compute rollout stats
                    main_rollout_stats = self._compute_rollout_stats(
                        main_batch, reward_extra_infos_dict, prefix="main"
                    )
                    metrics.update(main_rollout_stats)

                    # Log rollout data if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(
                            main_batch, reward_extra_infos_dict, timing_raw,
                            os.path.join(rollout_data_dir, "main"),
                        )

                # Validate
                if (
                    (self.val_reward_fn or self.use_reward_loop)
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Save checkpoint
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
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

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # Training metrics
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=main_batch, use_critic=False, agent_name="main"))
                metrics.update(compute_timing_metrics(batch=main_batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=main_batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return


class Phase1AgentLoopManager:
    """Agent loop manager for Phase 1 training.

    Uses Phase1AgentLoopWorker (extends AgentLoopWorkerBase) with optional reranker
    tool support via ToolModelManager.

    Key design:
        - Phase1AgentLoopWorker inherits from AgentLoopWorkerBase (same as train_search_r1_grpo.sh).
        - Adds optional reranker_server_handles and reranker_model_path for fixed reranker tool.
        - Reranker temperature is controlled by the tool config YAML (always 0.0).
        - Workers return single DataProto (no tuple unpacking needed).
        - No UID grouping, score assignment, or FSDP alignment logic.

    Supports two modes:
        - With reranker: tool_model_manager provides server handles + model path.
        - Retrieval-only: tool_model_manager=None, reranker_model_path=None.
    """

    def __init__(
        self,
        config,
        worker_group: RayWorkerGroup,
        tool_model_manager=None,
        reranker_model_path: str = None,
        rm_wg: RayWorkerGroup = None,
    ):
        """Initialize Phase 1 agent loop manager.

        Args:
            config: Trainer config.
            worker_group: Main agent worker group (ActorRollout).
            tool_model_manager: Optional ToolModelManager providing reranker server handles.
                If None, runs in retrieval-only mode.
            reranker_model_path: Optional HuggingFace model path for reranker tokenizer.
                Required if tool_model_manager is provided.
            rm_wg: Reward model worker group (optional).
        """
        from verl.experimental.agent_loop.phase1_agent_loop_worker import Phase1AgentLoopWorker
        from verl.workers.rollout.replica import get_rollout_replica_class

        self.config = config
        self.worker_group = worker_group
        self.tool_model_manager = tool_model_manager
        self.reranker_model_path = reranker_model_path

        # Reranker server handles (None if retrieval-only)
        self.reranker_server_handles = None
        if self.tool_model_manager is not None:
            self.reranker_server_handles = self.tool_model_manager.server_handles

        # Rule-based reward model (optional)
        self.reward_model_manager = None
        self.reward_router_address = None
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward import RewardModelManager
            self.reward_model_manager = RewardModelManager(config.reward_model, rm_wg)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        # Rollout replicas for main agent
        self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        self.agent_loop_workers_class = Phase1AgentLoopWorker

        # Initialize main agent vLLM servers (hybrid mode with worker_group)
        self._initialize_llm_servers()

        print(f"[Phase1AgentLoopManager] Main agent servers: {self.server_addresses}")
        if self.reranker_server_handles:
            print(f"[Phase1AgentLoopManager] Reranker tool servers: "
                  f"{self.tool_model_manager.server_addresses}")
        else:
            print("[Phase1AgentLoopManager] No reranker — retrieval-only mode")

        # Initialize agent loop workers
        self._init_agent_loop_workers()

        # Initially in sleep mode if configured
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        """Initialize main agent vLLM/SGLang servers (hybrid with FSDP worker group)."""
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

        # Prometheus
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False.")
            from verl.experimental.agent_loop.agent_loop import update_prometheus_config
            update_prometheus_config(rollout_config.prometheus, self.server_addresses)

    def _init_agent_loop_workers(self):
        """Initialize Phase1AgentLoopWorker instances with optional reranker tool."""
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"phase1_agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    config=self.config,
                    server_handles=self.server_handles,
                    reward_router_address=self.reward_router_address,
                    reranker_server_handles=self.reranker_server_handles,  # None if retrieval-only
                    reranker_model_path=self.reranker_model_path,  # None if retrieval-only
                )
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences. Returns single DataProto (main agent output only).

        Phase1AgentLoopWorker.generate_sequences returns DataProto directly
        (inherited from AgentLoopWorkerBase), no tuple unpacking needed.
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.wake_up()

        # Dispatch to workers
        chunks = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get([
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks, strict=True)
        ])

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.sleep()

        # Workers return DataProto directly — no tuple unpacking
        output = DataProto.concat(outputs)

        # Calculate performance metrics (same format as AgentLoopManager)
        metrics = [o.meta_info.pop("metrics") for o in outputs]
        timing, aggregated_metrics = self._performance_metrics(metrics, output)
        output.meta_info = {
            "timing": timing,
            "aggregated_metrics": aggregated_metrics,
            **outputs[0].meta_info,
        }

        return output

    def _performance_metrics(self, metrics: list, output: DataProto):
        """Compute performance metrics from worker metrics.

        Uses the same format as AgentLoopManager._performance_metrics.
        """
        timing = {}
        aggregated_metrics = {}

        # Flatten: metrics is List[List[Dict]] (one list per worker, one dict per sample)
        flat_metrics = [metric for chunk in metrics for metric in chunk]
        if not flat_metrics:
            return timing, aggregated_metrics

        # Timing stats
        t_generate_sequences = np.array([m.get("generate_sequences", 0) for m in flat_metrics])
        t_tool_calls = np.array([m.get("tool_calls", 0) for m in flat_metrics])
        timing["agent_loop/generate_sequences/min"] = float(t_generate_sequences.min())
        timing["agent_loop/generate_sequences/max"] = float(t_generate_sequences.max())
        timing["agent_loop/generate_sequences/mean"] = float(t_generate_sequences.mean())
        timing["agent_loop/tool_calls/min"] = float(t_tool_calls.min())
        timing["agent_loop/tool_calls/max"] = float(t_tool_calls.max())
        timing["agent_loop/tool_calls/mean"] = float(t_tool_calls.mean())

        # Slowest sample
        slowest = int(np.argmax(t_generate_sequences + t_tool_calls))
        if slowest < output.batch["attention_mask"].shape[0]:
            attention_mask = output.batch["attention_mask"][slowest]
            prompt_length = output.batch["prompts"].shape[1]
            timing["agent_loop/slowest/generate_sequences"] = float(t_generate_sequences[slowest])
            timing["agent_loop/slowest/tool_calls"] = float(t_tool_calls[slowest])
            timing["agent_loop/slowest/prompt_length"] = int(attention_mask[:prompt_length].sum().item())
            timing["agent_loop/slowest/response_length"] = int(attention_mask[prompt_length:].sum().item())

        # Aggregate reranker metrics if present
        if flat_metrics and "reranker_attempted" in flat_metrics[0]:
            total_samples = len(flat_metrics)
            num_attempted = sum(1 for m in flat_metrics if m.get("reranker_attempted", False))
            num_success = sum(1 for m in flat_metrics if m.get("reranker_success", False))
            num_fallback = sum(1 for m in flat_metrics if m.get("reranker_fallback", False))
            if total_samples > 0:
                aggregated_metrics["reranker/success_rate"] = num_success / total_samples
                aggregated_metrics["reranker/fallback_rate"] = num_fallback / total_samples
            retrieved_docs = [m.get("num_retrieved_docs", 0) for m in flat_metrics if "num_retrieved_docs" in m]
            reranked_docs = [m.get("num_reranked_docs", 0) for m in flat_metrics if "num_reranked_docs" in m]
            if retrieved_docs:
                aggregated_metrics["reranker/avg_retrieved_docs"] = float(np.mean(retrieved_docs))
            if reranked_docs:
                aggregated_metrics["reranker/avg_reranked_docs"] = float(np.mean(reranked_docs))

        return timing, aggregated_metrics

    def wake_up(self):
        """Wake up all rollout replica instances (main + optional reranker tool)."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])
        if self.tool_model_manager is not None:
            self.tool_model_manager.wake_up()

    def sleep(self):
        """Sleep all rollout replica instances (main + optional reranker tool)."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])
        if self.tool_model_manager is not None:
            self.tool_model_manager.sleep()

    def _run_all(self, tasks: list):
        """Run a list of async tasks synchronously."""
        async def _gather():
            return await asyncio.gather(*tasks)
        asyncio.run(_gather())

