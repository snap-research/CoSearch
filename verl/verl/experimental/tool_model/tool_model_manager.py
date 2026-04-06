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
"""
ToolModelManager — manages LLM models used as fixed tools in alternating training.

Architecture:
    - Follows the exact same pattern as LLMJudgeRewardModelManager.
    - No Role enum needed — creates standalone RayResourcePool + vLLM replicas.
    - Must be instantiated in init_workers() BEFORE FSDP training pools are created,
      so that it claims GPUs first and the training pools are forced onto remaining GPUs.

Usage:
    Phase 1: ToolModelManager manages a fixed Reranker vLLM server on Node 1.
    Phase 2: ToolModelManager manages a fixed Search-R1 vLLM server on Node 0 (half GPUs).
"""

import asyncio
import logging
import os
from typing import Optional

from verl.single_controller.ray.base import RayWorkerGroup
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import get_rollout_replica_class
from verl.utils.config import omega_conf_to_dataclass

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ToolModelManager:
    """Manager for LLM models used as fixed tools (not trained).

    This class is modeled after LLMJudgeRewardModelManager and follows the exact same
    initialization pattern:

    1. standalone mode: worker_group=None → creates its own RayResourcePool
    2. colocated mode: worker_group=<existing wg> → shares GPU with training workers

    In alternating training:
    - Phase 1: manages Reranker as a tool (standalone on Node 1 GPUs)
    - Phase 2: manages Search-R1 as a tool (standalone on Node 0 half GPUs)

    The manager provides:
    - server_handles: list of Ray actor handles for the vLLM servers
    - server_addresses: list of "ip:port" strings for HTTP access
    - router_address: single address for load-balanced access
    - tokenizer: the model's tokenizer (for prompt construction)
    - wake_up() / sleep(): GPU cache management
    """

    def __init__(self, config, worker_group: Optional[RayWorkerGroup] = None):
        """Initialize the tool model manager.

        Args:
            config: OmegaConf config with the following keys:
                - model.path: HuggingFace model path
                - model.external_lib: optional external lib
                - model.trust_remote_code: trust remote code flag
                - rollout.name: "vllm" or "sglang"
                - rollout.tensor_model_parallel_size: TP size
                - rollout.free_cache_engine: whether to sleep on init
                - n_gpus_per_node: GPUs per node for this tool
                - nnodes: number of nodes for this tool
            worker_group: Optional RayWorkerGroup for colocated mode.
                          If None, standalone mode is used (creates own GPUs).
        """
        self.config = config
        self.worker_group = worker_group
        self._initialize_llm_servers()
        self._initialize_router()
        if self.config.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        """Initialize vLLM/SGLang server replicas.

        In standalone mode: creates a new RayResourcePool and launches servers.
        In colocated mode: uses existing worker_group's GPUs.
        """
        rollout_world_size = self.config.rollout.tensor_model_parallel_size
        world_size = (
            self.worker_group.world_size
            if self.worker_group  # colocated mode
            else self.config.n_gpus_per_node * self.config.nnodes  # standalone mode
        )
        num_replicas = world_size // rollout_world_size

        rollout_replica_class = get_rollout_replica_class(self.config.rollout.name)

        # Convert raw OmegaConf to RolloutConfig dataclass
        rollout_config = omega_conf_to_dataclass(self.config.rollout, dataclass_type=RolloutConfig)
        model_config = HFModelConfig(
            path=self.config.model.path,
            external_lib=self.config.model.get("external_lib", None),
            trust_remote_code=self.config.model.get("trust_remote_code", True),
        )
        self.tokenizer = model_config.get_processor()

        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.n_gpus_per_node,
                is_reward_model=False,
                server_name_prefix="tool_",
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group:
            self._run_all([server.init_colocated(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        logger.info(
            f"[ToolModelManager] Initialized {num_replicas} tool server replicas "
            f"for model {self.config.model.path} at {self.server_addresses}"
        )

    def _initialize_router(self):
        """Initialize load-balancing router in front of server replicas."""
        worker_urls = [f"http://{addr}" for addr in self.server_addresses]

        if self.config.rollout.name == "sglang":
            from verl.experimental.reward.router.inner_sglang_router import launch_router_process
        else:
            from verl.experimental.reward.router.naive_router import launch_router_process

        self.router_address, _ = launch_router_process(worker_urls=worker_urls)
        logger.info(f"[ToolModelManager] Router initialized at {self.router_address}")

    def get_router_address(self) -> str:
        """Get the router address for HTTP-based access."""
        return self.router_address

    def wake_up(self):
        """Wake up all rollout replica instances (re-initialize KV cache)."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances (free KV cache memory)."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list):
        """Run a list of async tasks synchronously."""
        async def _gather():
            return await asyncio.gather(*tasks)
        return asyncio.run(_gather())
