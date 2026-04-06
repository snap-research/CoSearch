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
Dual Agent PPO training entry point for Search-R1 + Reranker system.
"""
import os
import socket
import warnings

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.search_r1_reranker_ray_trainer import CoSearchRayTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type


@hydra.main(config_path="config", config_name="co_search_trainer", version_base=None)
def main(config):
    """Main entry point for dual agent PPO training with Hydra configuration management."""
    run_dual_agent_ppo(config)


def run_dual_agent_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed dual agent PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            import verl.utils.transferqueue_utils as tqbridge_utils
            runtime_env_kwargs["worker_process_setup_hook"] = tqbridge_utils.worker_process_setup_hook

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(DualAgentTaskRunner)

    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class DualAgentTaskRunner:
    """Ray remote class for executing distributed dual agent PPO training tasks."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add main agent (Search-R1) actor rollout worker."""
        from verl.single_controller.ray import RayWorkerGroup

        if config.actor_rollout_ref.rollout.mode == "sync":
            warnings.warn("spmd rollout mode is deprecated and will be removed in v0.6.2", stacklevel=2)

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

        return actor_rollout_cls, ray_worker_group_cls

    def add_reranker_worker(self, config):
        """Add reranker agent worker based on trainable flag."""
        from verl.single_controller.ray import RayWorkerGroup

        if config.reranker_actor_rollout_ref.rollout.mode == "sync":
            warnings.warn("spmd rollout mode is deprecated and will be removed in v0.6.2", stacklevel=2)

        if config.reranker_actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            reranker_cls = (
                AsyncActorRolloutRefWorker
                if config.reranker_actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.reranker_actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            reranker_cls = (
                AsyncActorRolloutRefWorker
                if config.reranker_actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        # Check if reranker is trainable
        is_trainable = config.reranker_actor_rollout_ref.get("trainable", False)
        
        if is_trainable:
            # Use RerankerActorRollout for trainable reranker
            self.role_worker_mapping[Role.RerankerActorRollout] = ray.remote(reranker_cls)
        else:
            # Use RerankerRollout for inference-only reranker
            self.role_worker_mapping[Role.RerankerRollout] = ray.remote(reranker_cls)

        return reranker_cls, ray_worker_group_cls, is_trainable

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        if config.critic.enable:
            raise NotImplementedError("DualAgent system doesn't support critic yet.")

    def init_resource_pool_mgr(self, config, is_reranker_trainable):
        """Initialize resource pool manager.
        
        RESOURCE SPLIT ARCHITECTURE:
        Split GPUs into two independent pools to avoid memory contention.
        
        Single node (nnodes=1):
          - Split GPUs on the same node: first half for main agent, second half for reranker
          - Example: 8 GPUs → [4] for main, [4] for reranker
        
        Multi-node (nnodes>=2):
          - Split nodes: first half nodes for main agent, second half nodes for reranker
          - Example: 2 nodes with 8 GPUs each → [8, 0] for main, [0, 8] for reranker
          - Example: 4 nodes with 8 GPUs each → [8, 8, 0, 0] for main, [0, 0, 8, 8] for reranker
        """
        from verl.trainer.ppo.ray_trainer import Role

        nnodes = config.trainer.nnodes
        n_gpus_per_node = config.trainer.n_gpus_per_node
        total_gpus = n_gpus_per_node * nnodes
        
        main_agent_pool_id = "main_agent_pool"
        reranker_pool_id = "reranker_pool"
        
        if nnodes == 1:
            # Single node: split GPUs on the same node
            gpus_per_agent = n_gpus_per_node // 2
            main_agent_allocation = [gpus_per_agent]
            reranker_allocation = [gpus_per_agent]
            
            print(f"[Resource Split] Single node mode: {n_gpus_per_node} GPUs")
            print(f"[Resource Split] Main agent: GPUs 0-{gpus_per_agent-1}")
            print(f"[Resource Split] Reranker: GPUs {gpus_per_agent}-{n_gpus_per_node-1}")
        else:
            assert nnodes % 2 == 0, "Number of nodes (nnodes) must be even for multi-node resource split."
            # Multi-node: split nodes between agents
            nodes_per_agent = nnodes // 2
            
            # Main agent gets first half of nodes, reranker gets second half
            # Ray doesn't accept 0-value bundles, so we only include nodes with GPUs
            main_agent_allocation = [n_gpus_per_node] * nodes_per_agent
            reranker_allocation = [n_gpus_per_node] * (nnodes - nodes_per_agent)
            
            print(f"[Resource Split] Multi-node mode: {nnodes} nodes × {n_gpus_per_node} GPUs/node = {total_gpus} total GPUs")
            print(f"[Resource Split] Main agent: {nodes_per_agent} nodes with {n_gpus_per_node} GPUs each = {main_agent_allocation}")
            print(f"[Resource Split] Reranker: {nnodes - nodes_per_agent} nodes with {n_gpus_per_node} GPUs each = {reranker_allocation}")

        resource_pool_spec = {
            main_agent_pool_id: main_agent_allocation,
            reranker_pool_id: reranker_allocation,
        }

        # Reward model resource pool (if needed, use main agent pool or separate)
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        # Map roles to their respective resource pools
        # Main agent and its reference use main_agent_pool
        self.mapping[Role.ActorRollout] = main_agent_pool_id
        self.mapping[Role.RefPolicy] = main_agent_pool_id
        self.mapping[Role.Critic] = main_agent_pool_id  # If critic is used
        
        # Reranker and its reference use reranker_pool
        if is_reranker_trainable:
            self.mapping[Role.RerankerActorRollout] = reranker_pool_id
            self.mapping[Role.RerankerRefPolicy] = reranker_pool_id
        else:
            self.mapping[Role.RerankerRollout] = reranker_pool_id

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker
                else:
                    raise NotImplementedError
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import RewardModelWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker for main agent if KL loss or KL reward is used."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def add_reranker_ref_policy_worker(self, config, ref_policy_cls, is_trainable):
        """Add reference policy worker for reranker if trainable and KL is used."""
        from verl.trainer.ppo.ray_trainer import Role

        # Only add ref policy if reranker is trainable
        if not is_trainable:
            return

        if config.algorithm.use_kl_in_reward or config.reranker_actor_rollout_ref.actor.get("use_kl_loss", False):
            self.role_worker_mapping[Role.RerankerRefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RerankerRefPolicy] = "global_pool"

    def run(self, config):
        """Execute the main dual agent PPO training workflow."""
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"DualAgentTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Add main agent workers
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # Add reranker workers
        reranker_cls, _, is_reranker_trainable = self.add_reranker_worker(config)
        self.add_reranker_ref_policy_worker(config, reranker_cls, is_reranker_trainable)

        # Validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Download checkpoints
        trust_remote_code = config.data.get("trust_remote_code", False)
        
        # Main agent
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        
        # Reranker
        reranker_local_path = copy_to_local(
            config.reranker_actor_rollout_ref.model.path, use_shm=config.reranker_actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate tokenizers
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        
        reranker_tokenizer = hf_tokenizer(reranker_local_path, trust_remote_code=trust_remote_code)

        # Load reward manager
        if config.reward_model.use_reward_loop:
            reward_fn = None 
            val_reward_fn = None
        else:
            reward_fn = load_reward_manager(
                config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
            )
            val_reward_fn = load_reward_manager(
                config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
            )

        resource_pool_manager = self.init_resource_pool_mgr(config, is_reranker_trainable)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create datasets
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the dual agent PPO trainer
        trainer = CoSearchRayTrainer(
            config=config,
            tokenizer=tokenizer,
            reranker_tokenizer=reranker_tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            is_reranker_trainable=is_reranker_trainable,
        )
        
        # Initialize workers
        trainer.init_workers()

        # Start training
        trainer.fit()


if __name__ == "__main__":
    main()
