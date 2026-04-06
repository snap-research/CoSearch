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
import asyncio
import heapq
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional, List
from uuid import uuid4
import time

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.experimental.reward import RewardManagerWorker
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
    rollout_trace_op,
)
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def random_subsample_dataproto(data: DataProto, max_size: int) -> DataProto:
    """Randomly subsample a DataProto to a maximum size.
    
    Args:
        data: Input DataProto to subsample.
        max_size: Maximum number of samples to keep.
        
    Returns:
        Subsampled DataProto (modified in-place and returned).
    """
    if len(data) <= max_size:
        return data
    
    indices = torch.randperm(len(data))[:max_size].tolist()
    data.batch = data.batch[indices]
    for key in data.non_tensor_batch:
        data.non_tensor_batch[key] = data.non_tensor_batch[key][indices]
    return data


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput: token output
        """
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=uuid4().hex,  # use new request_id for each turn
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
        )
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


# make hydra.utils.instantiate happy
class _DummyConfig:
    def __init__(self, config: DictConfig) -> None:
        self.config = config


class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False

    def __init__(
        self,
        trainer_config: _DummyConfig,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (_DummyConfig): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
        """
        self.init_class(config=trainer_config.config, tokenizer=tokenizer, processor=processor, **kwargs)
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, processor: AutoProcessor, **kwargs):
        """This is used to do heavy initialization work that should shared across all instances. It's only called once.

        Args:
            config (DictConfig): trainer config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process multi_modal data.
            **kwargs: extra kwargs from config file passed in by `hydra.utils.instantiate`.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorkerBase:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config

        # for recipe to change
        if not hasattr(self, "server_manager"):
            self.server_manager = AsyncLLMServerManager(config, server_handles)

        self.reward_router_address = reward_router_address

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        self.reward_manager_worker = RewardManagerWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, self.reward_router_address)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

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
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs)
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)

            output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

            # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

            # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
            # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
            # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
            # input_ids: concatenation of prompt + response
            # Mask:
            # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
            # - prompt_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [0,0,0,0,1,1,1,1]
            # - response_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
            # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
            #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
            # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
            #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
            # - position_ids: sequential positions for tokens, starting at 0
            #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

            self.tokenizer.padding_side = "left"
            prompt_output = self.tokenizer.pad(
                {"input_ids": output.prompt_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

            self.tokenizer.padding_side = "right"
            response_output = self.tokenizer.pad(
                {"input_ids": output.response_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if response_output["input_ids"].dim() == 1:
                response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

            response_mask_output = self.tokenizer.pad(
                {"input_ids": output.response_mask},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if response_mask_output["input_ids"].dim() == 1:
                response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

            response_logprobs = None
            if output.response_logprobs is not None:
                pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
                response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

            response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

            # Handle multi-modal inputs and position_ids calculation
            # Only support Qwen2VLImageProcessor for multi-modal processing currently
            # TODO: support other multi-modal inputs
            multi_modal_inputs = None
            if (
                self.processor is not None
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                from verl.models.transformers.qwen2_vl import get_rope_index

                images = getattr(output, "multi_modal_data", {}).get("image", None)
                current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
                multi_modal_inputs.pop("input_ids", None)
                multi_modal_inputs.pop("attention_mask", None)

                # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
                # because np.array() only keeps the keys for BatchFeature.
                multi_modal_inputs = dict(multi_modal_inputs)

                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids.squeeze(0),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask.squeeze(0),
                ).unsqueeze(0)  # (1, 3, seq_len)

                valid_mask = attention_mask[0].bool()
                text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                text_position_ids = text_position_ids.unsqueeze(0)
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)
            
            # Enable async reward computation if:
            # 1. Reward router is available (enable_resource_pool=True), OR
            # 2. Reward model is disabled (let ray_trainer handle it instead)
            enable_async_reward = (
                self.reward_router_address is not None and self.config.reward_model.enable_resource_pool
            )  or self.config.reward_model.use_reward_loop # modify based on https://github.com/volcengine/verl/issues/4346
        
            
            if output.reward_score is None and enable_async_reward:
                batch = TensorDict(
                    {
                        "prompts": prompt_output["input_ids"],  # [1, prompt_length]
                        "responses": response_output["input_ids"],  # [1, response_length]
                        "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                        "input_ids": input_ids,  # [1, prompt_length + response_length]
                        "position_ids": position_ids,
                    },
                    batch_size=1,
                )
                non_tensor_batch = {
                    **{k: np.array([v]) for k, v in kwargs.items()},
                    "__num_turns__": np.array([output.num_turns]),
                    "tool_extra_fields": np.array([output.extra_fields], dtype=object),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                
                result = await self.reward_manager_worker.compute_score.remote(data)
                
                output.reward_score = result["reward_score"]
                output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

            return _InternalAgentLoopOutput(
                prompt_ids=prompt_output["input_ids"],
                response_ids=response_output["input_ids"],
                input_ids=input_ids,
                position_ids=position_ids,
                response_mask=response_mask,
                attention_mask=attention_mask,
                response_logprobs=response_logprobs,
                multi_modal_inputs=multi_modal_inputs,
                multi_modal_data=output.multi_modal_data,
                reward_score=output.reward_score,
                num_turns=output.num_turns,
                metrics=output.metrics,
                extra_fields=output.extra_fields,
            )

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )

    def create_transferqueue_client(self, controller_infos, storage_infos, role):
        """Create a client for data system(transfer queue)."""
        from verl.single_controller.ray.base import get_random_string
        from verl.utils.transferqueue_utils import create_transferqueue_client

        client_name = get_random_string(length=6)
        create_transferqueue_client(
            client_id=f"{role}_worker_{client_name}",
            controller_infos=controller_infos,
            storage_infos=storage_infos,
        )


@ray.remote
class AgentLoopWorker(AgentLoopWorkerBase):
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None
    ):
        """Initialize agent loop manager.
        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            reward_router_address (str): reward router address.
        """
        super().__init__(config, server_handles, reward_router_address)


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group for hybrid mode; None for standalone mode.
        """
        self.config = config
        self.worker_group = worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward import RewardModelManager

            self.reward_model_manager = RewardModelManager(config.reward_model, rm_wg)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = AgentLoopWorker

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
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
        if self.worker_group:
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"AgentLoopManager: Main agent servers initialized at {self.server_addresses}")

        # Update Prometheus configuration with server addresses
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(rollout_config.prometheus, self.server_addresses)

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.reward_router_address)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.wake_up()

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.sleep()

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing, aggregated_metrics = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, "aggregated_metrics": aggregated_metrics, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> tuple[dict[str, float], dict[str, Any]]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        # Aggregate reranker metrics if present
        aggregated_metrics = {}
        flat_metrics = [metric for chunk in metrics for metric in chunk]
        
        # Check if any sample has reranker metrics
        if flat_metrics and "reranker_attempted" in flat_metrics[0]:
            total_samples = len(flat_metrics)
            num_attempted = sum(1 for m in flat_metrics if m.get("reranker_attempted", False))
            num_success = sum(1 for m in flat_metrics if m.get("reranker_success", False))
            num_fallback = sum(1 for m in flat_metrics if m.get("reranker_fallback", False))
            
            # Count by fallback reason
            num_format_errors = sum(1 for m in flat_metrics 
                                   if m.get("reranker_fallback_reason") == "format_validation_error")
            num_execution_errors = sum(1 for m in flat_metrics 
                                      if m.get("reranker_fallback_reason") == "execution_error")
            
            # Aggregated metrics
            aggregated_metrics["reranker/attempted_count"] = num_attempted
            aggregated_metrics["reranker/success_count"] = num_success
            aggregated_metrics["reranker/fallback_count"] = num_fallback
            aggregated_metrics["reranker/format_error_count"] = num_format_errors
            aggregated_metrics["reranker/execution_error_count"] = num_execution_errors
            
            # Rates (avoid division by zero)
            if total_samples > 0:
                aggregated_metrics["reranker/success_rate"] = num_success / total_samples
                aggregated_metrics["reranker/fallback_rate"] = num_fallback / total_samples
                aggregated_metrics["reranker/format_error_rate"] = num_format_errors / total_samples
                aggregated_metrics["reranker/execution_error_rate"] = num_execution_errors / total_samples
            
            # Average document counts
            retrieved_docs = [m.get("num_retrieved_docs", 0) for m in flat_metrics if "num_retrieved_docs" in m]
            reranked_docs = [m.get("num_reranked_docs", 0) for m in flat_metrics if "num_reranked_docs" in m]
            if retrieved_docs:
                aggregated_metrics["reranker/avg_retrieved_docs"] = np.mean(retrieved_docs)
            if reranked_docs:
                aggregated_metrics["reranker/avg_reranked_docs"] = np.mean(reranked_docs)

        return timing, aggregated_metrics

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())


# ---------------------- My Modifications for Search-R1 with dual agent support ---------------------- #
class SearchR1DualAgentLoopWorkerBase(AgentLoopWorkerBase):
    """Agent loop worker with dual agent support (main agent + reranker).
    
    This worker extends AgentLoopWorker to support reranker integration:
    - Accepts reranker_server_handles parameter
    - Creates reranker_server_manager for reranker agent calls
    - Loads reranker_tokenizer (may be different from main tokenizer)
    - Passes reranker parameters to agent loops (e.g., SearchR1AgentLoop)
    """
    
    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reranker_server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
    ):
        """Initialize dual agent loop worker.
        
        Args:
            config: YAML config.
            server_handles: Main agent server actor handles.
            reward_router_address: Reward router address.
            reranker_server_handles: Optional reranker server actor handles.
        """
        # Call parent constructor
        super().__init__(config, server_handles, reward_router_address)

        if not hasattr(self, "reranker_server_manager"): 
            self.reranker_server_manager = AsyncLLMServerManager(config, reranker_server_handles)

        model_path = config.reranker_actor_rollout_ref.model.path
        self.reranker_model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.reranker_actor_rollout_ref.model.path)
        self.reranker_tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.reranker_processor = hf_processor(local_path, trust_remote_code=True)
        
        print("main model path:", config.actor_rollout_ref.model.path)
        print("reranker model path:", model_path)
    
    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        """
        Extend from _run_agent_loop in AgentLoopWorker to pass reranker parameters to agent loops.
        """

        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            track_messages = True if self.config.reranker_actor_rollout_ref.trainable else False
            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                reranker_server_manager=self.reranker_server_manager,
                reranker_tokenizer=self.reranker_tokenizer,
                tokenizer=self.tokenizer,
                processor=self.processor,
                track_messages=track_messages,
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)

            output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]
            
            # Preserve uid from kwargs if it exists (needed for GRPO grouping)
            if "uid" in kwargs:
                output.extra_fields["uid"] = kwargs["uid"]
            # Preserve reward_model for logging ground truth
            if "reward_model" in kwargs:
                output.extra_fields["reward_model"] = kwargs["reward_model"]
            
            # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

            # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
            # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
            # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
            # input_ids: concatenation of prompt + response
            # Mask:
            # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
            # - prompt_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [0,0,0,0,1,1,1,1]
            # - response_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
            # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
            #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
            # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
            #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
            # - position_ids: sequential positions for tokens, starting at 0
            #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

            self.tokenizer.padding_side = "left"
            prompt_output = self.tokenizer.pad(
                {"input_ids": output.prompt_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

            self.tokenizer.padding_side = "right"
            response_output = self.tokenizer.pad(
                {"input_ids": output.response_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if response_output["input_ids"].dim() == 1:
                response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

            response_mask_output = self.tokenizer.pad(
                {"input_ids": output.response_mask},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if response_mask_output["input_ids"].dim() == 1:
                response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

            response_logprobs = None
            if output.response_logprobs is not None:
                pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
                response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

            response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

            # Handle multi-modal inputs and position_ids calculation
            # Only support Qwen2VLImageProcessor for multi-modal processing currently
            # TODO: support other multi-modal inputs
            multi_modal_inputs = None
            if (
                self.processor is not None
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                from verl.models.transformers.qwen2_vl import get_rope_index

                images = getattr(output, "multi_modal_data", {}).get("image", None)
                current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
                multi_modal_inputs.pop("input_ids", None)
                multi_modal_inputs.pop("attention_mask", None)

                # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
                # because np.array() only keeps the keys for BatchFeature.
                multi_modal_inputs = dict(multi_modal_inputs)

                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids.squeeze(0),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask.squeeze(0),
                ).unsqueeze(0)  # (1, 3, seq_len)

                valid_mask = attention_mask[0].bool()
                text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                text_position_ids = text_position_ids.unsqueeze(0)
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)
            
            # Enable async reward computation if:
            # 1. Reward router is available (enable_resource_pool=True), OR
            # 2. Reward model is disabled (let ray_trainer handle it instead)
            enable_async_reward = (
                self.reward_router_address is not None and self.config.reward_model.enable_resource_pool
            )  or self.config.reward_model.use_reward_loop # modify based on https://github.com/volcengine/verl/issues/4346
        
            
            if output.reward_score is None and enable_async_reward:
                batch = TensorDict(
                    {
                        "prompts": prompt_output["input_ids"],  # [1, prompt_length]
                        "responses": response_output["input_ids"],  # [1, response_length]
                        "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                        "input_ids": input_ids,  # [1, prompt_length + response_length]
                        "position_ids": position_ids,
                    },
                    batch_size=1,
                )
                non_tensor_batch = {
                    **{k: np.array([v]) for k, v in kwargs.items()},
                    "__num_turns__": np.array([output.num_turns]),
                    "tool_extra_fields": np.array([output.extra_fields], dtype=object),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                
                result = await self.reward_manager_worker.compute_score.remote(data)
                
                output.reward_score = result["reward_score"]
                output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

            return _InternalAgentLoopOutput(
                prompt_ids=prompt_output["input_ids"],
                response_ids=response_output["input_ids"],
                input_ids=input_ids,
                position_ids=position_ids,
                response_mask=response_mask,
                attention_mask=attention_mask,
                response_logprobs=response_logprobs,
                multi_modal_inputs=multi_modal_inputs,
                multi_modal_data=output.multi_modal_data,
                reward_score=output.reward_score,
                num_turns=output.num_turns,
                metrics=output.metrics,
                extra_fields=output.extra_fields,
            )

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        """copy from AgentLoopWorkerBase, but we let uid in extra_fields, and make sure they will
        be in non_tensor_batch["uid"], which will be used for GRPO.
        """
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)
        
        # self._debug_postprocess_print(batch, non_tensor_batch, reward_extra_keys, metrics)

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )

    def _debug_postprocess_print(self, batch, non_tensor_batch, reward_extra_keys, metrics):
        # Print batch info for debugging
        print("\n" + "="*80)
        print("SearchR1DualAgentLoopWorkerBase._postprocess - Batch Information")
        print("="*80)
        print(f"\n[Tensor Batch] keys and shapes:")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        print(f"\n[Non-Tensor Batch] keys and shapes:")
        for key, value in non_tensor_batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, np.ndarray):
                print(f"  {key}: np.ndarray(shape={value.shape}, dtype={value.dtype})")
            else:
                print(f"  {key}: {type(value)} (len={len(value) if hasattr(value, '__len__') else 'N/A'})")
        
        print(f"\n[Meta Info] keys:")
        print(f"  reward_extra_keys: {reward_extra_keys}")
        print(f"  metrics: list of {len(metrics)} items")
        print("="*80 + "\n")


class SearchR1DualAgentLoopManager(AgentLoopManager):
    """Agent loop manager for Search-R1 with dual agent support (main agent + reranker).
    
    This manager handles two sets of vLLM/SGLang servers:
    - Main agent server: For Search-R1 generation
    - Reranker server: For document reranking
    
    It supports two training modes:
    - train_mode="search_r1": Train main agent, reranker is fixed (deterministic)
    - train_mode="reranker": Train reranker with counterfactual rollout, main agent is fixed
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        reranker_worker_group: RayWorkerGroup = None,
        rm_wg: RayWorkerGroup = None,
    ):
        """Initialize SearchR1 dual agent loop manager.
        
        Args:
            config: Trainer config.
            worker_group: Main agent worker group (ActorRolloutRef).
            reranker_worker_group: Reranker worker group.
            rm_wg: Reward model worker group.
        """
        # Set the worker class to ExtendedAgentLoopWorker before calling parent __init__
        from verl.experimental.agent_loop.search_r1_dual_agent_loop import SearchR1DualAgentLoopWorker

        self.config = config
        self.worker_group = worker_group
        self.reranker_worker_group = reranker_worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward import RewardModelManager

            self.reward_model_manager = RewardModelManager(config.reward_model, rm_wg)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        self.agent_loop_workers_class = SearchR1DualAgentLoopWorker

        self._initialize_llm_servers()
        # Initialize reranker servers (only if reranker_worker_group is provided)
        if self.reranker_worker_group is not None:
            self._initialize_reranker_lllm_servers()

        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        self.train_rr_main_ratio = config.get("train_rr_main_ratio", 2)

    
    def _initialize_reranker_lllm_servers(self):
        """Initialize reranker vLLM/SGLang servers."""
        print(f"[SearchR1DualAgent] Starting reranker server initialization...")
        
        # Calculate reranker server configuration
        rollout_world_size = (
            self.config.reranker_actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.reranker_actor_rollout_ref.rollout.data_parallel_size
            * self.config.reranker_actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = self.reranker_worker_group.world_size
        num_replicas = world_size // rollout_world_size
        
        rollout_config = self.config.reranker_actor_rollout_ref.rollout
        model_config = self.config.reranker_actor_rollout_ref.model
        
        # Create reranker replicas with unique name prefix to avoid conflicts with main agent
        self.reranker_rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
                server_name_prefix="reranker_",  # Prefix to distinguish from main agent servers
            )
            for replica_rank in range(num_replicas)
        ]
        
        print(f"[SearchR1DualAgent] Initializing {num_replicas} reranker server replicas...")
        # Initialize reranker servers in hybrid mode
        self._run_all([server.init_hybrid(self.reranker_worker_group) for server in self.reranker_rollout_replicas])
        
        self.reranker_server_handles = [server._server_handle for server in self.reranker_rollout_replicas]
        self.reranker_server_addresses = [server._server_address for server in self.reranker_rollout_replicas]
        
        print(f"[SearchR1DualAgent] Reranker servers initialized at {self.reranker_server_addresses}")
        
        # Update Prometheus for reranker if enabled
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False for reranker.")
            update_prometheus_config(rollout_config.prometheus, self.reranker_server_addresses)
    
    def _init_agent_loop_workers(self):
        """Initialize agent loop workers with dual agent support."""
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"search_r1_dual_agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.server_handles,
                    self.reranker_server_handles,
                    self.reward_router_address,
                )
            )
    
    def generate_sequences(self, prompts: DataProto, enable_reranker_training: bool = False) -> DataProto | tuple[DataProto, DataProto]:
        """Generate sequences with support for different training modes.
        
        Args:
            prompts: Input batch.
            enable_reranker_training: If True, use counterfactual rollout for dual agent training.
            
        Returns:
            - If enable_reranker_training=False: Single DataProto for normal generation
            - If enable_reranker_training=True: Tuple of (main_agent_batch, reranker_agent_batch)
        """
        if enable_reranker_training:
            return self._generate_train_sequences(prompts)
        else:
            return self._generate_sequences(prompts)
            

    def _generate_sequences(self, prompts: DataProto) -> DataProto:
        """Normal generation for training search-r1 agent.
        
        - Main agent is trainable (temperature > 0)
        - Reranker is fixed (deterministic, temperature=0)
        - Standard agent loop execution
        """
        # assert prompts.meta_info["validate"] is True, "Normal generation only supports validate=True mode."
        return super().generate_sequences(prompts)
    
    def _generate_train_sequences(self, prompts: DataProto) -> tuple[DataProto, DataProto]:
        """Special generation for training reranker with counterfactual rollouts.
        
        Workflow:
        1. For each prompt, run initial search_r1 rollout
        2. Identify all tool call positions as branch points
        3. For each branch point:
           - Sample 4 different reranker responses (temperature > 0)
           - Continue generation from each branch (main agent fixed)
        4. Group branches by branch point for GRPO
        5. Compute rewards based on final answers
        
        Args:
            prompts: Input batch (N questions × M rollouts).
            
        Returns:
            Tuple of (main_agent_batch, reranker_agent_batch).
        """
        logger.info(f"Training reranker with counterfactual rollout on {len(prompts)} prompts")
        
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.wake_up()
        
        # Run counterfactual rollout on each worker
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get([
            worker.generate_sequences_counterfactual.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
        ])
        
        # outputs is a list of tuples: [(main_batch_worker0, reranker_batch_worker0), ...]
        # Separate main and reranker outputs
        main_outputs = [output[0] for output in outputs]
        reranker_outputs = [output[1] for output in outputs]
        
        main_output = DataProto.concat(main_outputs)
        reranker_output = DataProto.concat(reranker_outputs)
        
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.sleep()
        
        # Calculate performance metrics for main agent
        main_metrics = [output.meta_info.pop("metrics") for output in main_outputs]
        main_timing, main_aggregated_metrics = self._performance_metrics(main_metrics, main_output)
        main_output.meta_info = {"timing": main_timing, "aggregated_metrics": main_aggregated_metrics, **main_outputs[0].meta_info}
        
        # Calculate performance metrics for reranker
        reranker_metrics = [output.meta_info.pop("metrics") for output in reranker_outputs]
        reranker_timing, reranker_aggregated_metrics = self._performance_metrics(reranker_metrics, reranker_output)
        reranker_output.meta_info = {"timing": reranker_timing, "aggregated_metrics": reranker_aggregated_metrics, **reranker_outputs[0].meta_info}
        
        return main_output, reranker_output
    
    def wake_up(self):
        """Wake up all rollout replica instances (main + reranker)."""
        super().wake_up()
        if hasattr(self, 'reranker_rollout_replicas'):
            self._run_all([replica.wake_up() for replica in self.reranker_rollout_replicas])
    
    def sleep(self):
        """Sleep all rollout replica instances (main + reranker)."""
        super().sleep()
        if hasattr(self, 'reranker_rollout_replicas'):
            self._run_all([replica.sleep() for replica in self.reranker_rollout_replicas])


# ----------------------- My moficiations for Search with reranker agent support ----------------------- #
class SearchR1RerankerAgentLoopWorkerBase(AgentLoopWorkerBase):
    """Agent loop worker with dual agent support (main agent + reranker).
    
    This worker extends AgentLoopWorker to support reranker integration:
    - Accepts reranker_server_handles parameter
    - Creates reranker_server_manager for reranker agent calls
    - Loads reranker_tokenizer (may be different from main tokenizer)
    - Passes reranker parameters to agent loops (e.g., SearchR1AgentLoop)
    """
    
    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reranker_server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
    ):
        """Initialize dual agent loop worker.
        
        Args:
            config: YAML config.
            server_handles: Main agent server actor handles.
            reward_router_address: Reward router address.
            reranker_server_handles: Optional reranker server actor handles.
        """
        # Call parent constructor
        super().__init__(config, server_handles, reward_router_address)

        if not hasattr(self, "reranker_server_manager"): 
            self.reranker_server_manager = AsyncLLMServerManager(config, reranker_server_handles)

        model_path = config.reranker_actor_rollout_ref.model.path
        self.reranker_model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.reranker_actor_rollout_ref.model.path)
        self.reranker_tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.reranker_processor = hf_processor(local_path, trust_remote_code=True)
        
        # Reward server manager (LLM-as-Judge) - None by default
        # Subclass SearchR1RerankerRewardAgentLoopWorkerBase overrides these
        if not hasattr(self, "reward_server_manager"):
            self.reward_server_manager = None
        if not hasattr(self, "reward_tokenizer"):
            self.reward_tokenizer = None
        
        print("main model path:", config.actor_rollout_ref.model.path)
        print("reranker model path:", model_path)
    
    def _convert_reranker_agent_output_to_internal(
        self,
        output: AgentLoopOutput,
        tokenizer: Any,
        prompt_length: int,
        response_length: int,
    ) -> _InternalAgentLoopOutput:
        """Convert AgentLoopOutput to _InternalAgentLoopOutput for reranker outputs.
        
        This helper function handles padding and mask generation for text-only outputs
        without multi-modal data or reward computation.
        
        Args:
            output: AgentLoopOutput from agent loop
            tokenizer: Tokenizer to use for padding (e.g., reranker_tokenizer)
            prompt_length: Max prompt length for padding
            response_length: Max response length for padding
            
        Returns:
            _InternalAgentLoopOutput with padded tensors
        """

        # Pad prompt (left padding)
        tokenizer.padding_side = "left"
        prompt_output = tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        # Pad response (right padding)
        tokenizer.padding_side = "right"
        response_output = tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        # Pad response mask
        response_mask_output = tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        # Pad response logprobs if present
        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        # Compute masks and concatenate
        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)
        
        # Compute position_ids (text-only, no multi-modal data)
        position_ids = compute_position_id_with_mask(attention_mask)

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            multi_modal_inputs=None,  # No multi-modal data
            multi_modal_data=output.multi_modal_data,
            reward_score=output.extra_fields.get("final_score", 0.0),
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )
    
    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> [_InternalAgentLoopOutput, list[AgentLoopOutput]]:
        """
        Extend from _run_agent_loop in AgentLoopWorker to pass reranker parameters to agent loops.
        """

        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            track_messages = True if self.config.reranker_actor_rollout_ref.trainable else False
            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                reranker_server_manager=self.reranker_server_manager,
                reranker_tokenizer=self.reranker_tokenizer,
                reward_server_manager=self.reward_server_manager,
                reward_tokenizer=self.reward_tokenizer,
                reward_http_client=getattr(self, "reward_http_client", None),  # HTTP mode support
                judge_semaphore=getattr(self, "_judge_semaphore", None),  # shared worker-level semaphore
                tokenizer=self.tokenizer,
                processor=self.processor,
                track_messages=track_messages,
            )
            # AgentLoopOutput and List[AgentLoopOoutput]
            output, reranker_outputs = await agent_loop.run(sampling_params, **kwargs)

            output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]
            
            
            # Preserve uid from kwargs if it exists (needed for GRPO grouping)
            assert "uid" in output.extra_fields, "uid must be present in output.extra_fields"
            
            if "reward_model" in kwargs:
                output.extra_fields["reward_model"] = kwargs["reward_model"]
            
            # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

            # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
            # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
            # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
            # input_ids: concatenation of prompt + response
            # Mask:
            # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
            # - prompt_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [0,0,0,0,1,1,1,1]
            # - response_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
            # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
            #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
            # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
            #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
            # - position_ids: sequential positions for tokens, starting at 0
            #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

            self.tokenizer.padding_side = "left"
            prompt_output = self.tokenizer.pad(
                {"input_ids": output.prompt_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

            self.tokenizer.padding_side = "right"
            response_output = self.tokenizer.pad(
                {"input_ids": output.response_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if response_output["input_ids"].dim() == 1:
                response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

            response_mask_output = self.tokenizer.pad(
                {"input_ids": output.response_mask},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if response_mask_output["input_ids"].dim() == 1:
                response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

            response_logprobs = None
            if output.response_logprobs is not None:
                pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
                response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

            response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

            # Handle multi-modal inputs and position_ids calculation
            # Only support Qwen2VLImageProcessor for multi-modal processing currently
            # TODO: support other multi-modal inputs
            multi_modal_inputs = None
            if (
                self.processor is not None
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                from verl.models.transformers.qwen2_vl import get_rope_index

                images = getattr(output, "multi_modal_data", {}).get("image", None)
                current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
                multi_modal_inputs.pop("input_ids", None)
                multi_modal_inputs.pop("attention_mask", None)

                # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
                # because np.array() only keeps the keys for BatchFeature.
                multi_modal_inputs = dict(multi_modal_inputs)

                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids.squeeze(0),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask.squeeze(0),
                ).unsqueeze(0)  # (1, 3, seq_len)

                valid_mask = attention_mask[0].bool()
                text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                text_position_ids = text_position_ids.unsqueeze(0)
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)
            
            # Enable async reward computation if:
            # 1. Reward router is available (enable_resource_pool=True), OR
            # 2. Reward model is disabled (let ray_trainer handle it instead)
            enable_async_reward = (
                self.reward_router_address is not None and self.config.reward_model.enable_resource_pool
            )  or self.config.reward_model.use_reward_loop # modify based on https://github.com/volcengine/verl/issues/4346
        
            
            if output.reward_score is None and enable_async_reward:
                batch = TensorDict(
                    {
                        "prompts": prompt_output["input_ids"],  # [1, prompt_length]
                        "responses": response_output["input_ids"],  # [1, response_length]
                        "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                        "input_ids": input_ids,  # [1, prompt_length + response_length]
                        "position_ids": position_ids,
                    },
                    batch_size=1,
                )
                non_tensor_batch = {
                    **{k: np.array([v]) for k, v in kwargs.items()},
                    "__num_turns__": np.array([output.num_turns]),
                    "tool_extra_fields": np.array([output.extra_fields], dtype=object),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                
                result = await self.reward_manager_worker.compute_score.remote(data)
                
                output.reward_score = result["reward_score"]
                output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

            for rr_out in reranker_outputs:
                rr_out.extra_fields["agent_score"] = output.reward_score 
                rr_out.extra_fields["golden_answer"] = output.extra_fields["reward_model"]["ground_truth"]
            
            return _InternalAgentLoopOutput(
                prompt_ids=prompt_output["input_ids"],
                response_ids=response_output["input_ids"],
                input_ids=input_ids,
                position_ids=position_ids,
                response_mask=response_mask,
                attention_mask=attention_mask,
                response_logprobs=response_logprobs,
                multi_modal_inputs=multi_modal_inputs,
                multi_modal_data=output.multi_modal_data,
                reward_score=output.reward_score,
                num_turns=output.num_turns,
                metrics=output.metrics,
                extra_fields=output.extra_fields,
            ), reranker_outputs

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        """copy from AgentLoopWorkerBase, but we let uid in extra_fields, and make sure they will
        be in non_tensor_batch["uid"], which will be used for GRPO.
        """
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)
        
        # self._debug_postprocess_print(batch, non_tensor_batch, reward_extra_keys, metrics)

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )
    

class SearchR1RerankerRewardAgentLoopWorkerBase(SearchR1RerankerAgentLoopWorkerBase):
    """Agent loop worker with triple agent support (main + reranker + reward LLM judge).
    
    Extends SearchR1RerankerAgentLoopWorkerBase to add:
    - reward_server_manager: AsyncLLMServerManager for LLM-as-Judge (Ray mode)
    - reward_http_client: RewardJudgeHttpServerManager for external server (HTTP mode)
    - reward_tokenizer: tokenizer for reward model
    
    Supports two modes:
    - Ray mode (2-node): Reward judge runs as Ray-managed vLLM server
    - HTTP mode (3-node): Reward judge runs as external HTTP server on separate VM
    """
    
    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reranker_server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
        reward_server_handles: list[ray.actor.ActorHandle] = None,
        reward_http_server_urls: list[str] = None,
    ):
        """Initialize triple agent loop worker.
        
        Args:
            config: YAML config.
            server_handles: Main agent server actor handles.
            reranker_server_handles: Reranker server actor handles.
            reward_router_address: Reward router address (for rule-based reward).
            reward_server_handles: Reward model (LLM-as-Judge) server actor handles (Ray mode).
            reward_http_server_urls: List of external reward judge server URLs (HTTP mode).
        """
        # Determine reward judge mode from config
        judge_config = config.get("reward_judge_model", {})
        reward_judge_mode = judge_config.get("mode", "ray")
        
        # Initialize HTTP client for external reward judge (3-node architecture)
        self.reward_http_client = None
        if reward_http_server_urls or reward_judge_mode == "http_server":
            from verl.experimental.agent_loop.reward_judge_http_client import RewardJudgeHttpServerManager
            
            server_urls = reward_http_server_urls or judge_config.get("server_urls", ["http://localhost:8000"])
            if isinstance(server_urls, str):
                server_urls = [server_urls]
            self.reward_http_client = RewardJudgeHttpServerManager(config, server_urls=server_urls)
            self.reward_server_manager = None
            self.reward_tokenizer = None  # Not needed for HTTP mode
            print(f"[RewardWorkerBase] Initialized HTTP client for reward judge: {server_urls}")
        
        # Initialize Ray-managed reward server (2-node architecture)
        elif reward_server_handles:
            self.reward_server_manager = AsyncLLMServerManager(config, reward_server_handles)
            # Load reward model tokenizer
            reward_model_path = config.reward_judge_model.model.path
            local_path = copy_to_local(reward_model_path)
            self.reward_tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
            self.reward_http_client = None
            print(f"[RewardWorkerBase] Initialized Ray reward server with {len(reward_server_handles)} handles")
            print(f"[RewardWorkerBase] Reward model path: {reward_model_path}")
        else:
            self.reward_server_manager = None
            self.reward_tokenizer = None
            self.reward_http_client = None
            print("[RewardWorkerBase] No reward judge backend configured; LLM-as-Judge disabled")
        
        # Create a SHARED judge semaphore at the Worker level.
        # This single semaphore is passed to ALL AgentLoop instances created by this worker,
        # so it limits concurrency ACROSS all samples (not per-sample like before).
        # With 8 workers × 64 = 512 max concurrent, matching 2 servers × 256 max_num_seqs.
        self.judge_max_concurrency = judge_config.get("judge_max_concurrency", 64)
        self._judge_semaphore = asyncio.Semaphore(self.judge_max_concurrency)
        print(f"[RewardWorkerBase] Shared judge semaphore: max_concurrency={self.judge_max_concurrency} (per worker)")
        
        # Call parent (SearchR1RerankerAgentLoopWorkerBase)
        super().__init__(config, server_handles, reranker_server_handles, reward_router_address)


class CoSearchAgentLoopManager(AgentLoopManager):
    """Agent loop manager for CoSearch with dual agent support (main agent + reranker).

    This manager handles two sets of vLLM/SGLang servers:
    - Main agent server: For CoSearch main agent generation
    - Reranker server: For document reranking

    It supports two training modes:
    - train_mode="search_r1": Train main agent, reranker is fixed (deterministic)
    - train_mode="reranker": Train reranker with counterfactual rollout, main agent is fixed
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        reranker_worker_group: RayWorkerGroup = None,
        rm_wg: RayWorkerGroup = None,
    ):
        """Initialize SearchR1 dual agent loop manager.
        
        Args:
            config: Trainer config.
            worker_group: Main agent worker group (ActorRolloutRef).
            reranker_worker_group: Reranker worker group.
            rm_wg: Reward model worker group.
        """
        # Import the correct worker class for CoSearch
        from verl.experimental.agent_loop.search_r1_reranker_agent_loop_worker import CoSearchAgentLoopWorker

        self.config = config
        self.worker_group = worker_group
        self.reranker_worker_group = reranker_worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward import RewardModelManager

            self.reward_model_manager = RewardModelManager(config.reward_model, rm_wg)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        # Set the correct worker class - CoSearchAgentLoopWorker, not SearchR1DualAgentLoopWorker!
        self.agent_loop_workers_class = CoSearchAgentLoopWorker

        self._initialize_llm_servers()
        # Initialize reranker servers (only if reranker_worker_group is provided)
        if self.reranker_worker_group is not None:
            self._initialize_reranker_lllm_servers()

        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        self.train_rr_main_ratio = config.get("train_rr_main_ratio", 2)

    
    def _initialize_reranker_lllm_servers(self):
        """Initialize reranker vLLM/SGLang servers."""
        print(f"[SearchR1DualAgent] Starting reranker server initialization...")
        
        # Calculate reranker server configuration
        rollout_world_size = (
            self.config.reranker_actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.reranker_actor_rollout_ref.rollout.data_parallel_size
            * self.config.reranker_actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = self.reranker_worker_group.world_size
        num_replicas = world_size // rollout_world_size
        
        rollout_config = self.config.reranker_actor_rollout_ref.rollout
        model_config = self.config.reranker_actor_rollout_ref.model
        
        # Create reranker replicas with unique name prefix to avoid conflicts with main agent
        self.reranker_rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
                server_name_prefix="reranker_",  # Prefix to distinguish from main agent servers
            )
            for replica_rank in range(num_replicas)
        ]
        
        print(f"[SearchR1DualAgent] Initializing {num_replicas} reranker server replicas...")
        # Initialize reranker servers in hybrid mode
        self._run_all([server.init_hybrid(self.reranker_worker_group) for server in self.reranker_rollout_replicas])
        
        self.reranker_server_handles = [server._server_handle for server in self.reranker_rollout_replicas]
        self.reranker_server_addresses = [server._server_address for server in self.reranker_rollout_replicas]
        
        print(f"[SearchR1DualAgent] Reranker servers initialized at {self.reranker_server_addresses}")
        
        # Update Prometheus for reranker if enabled
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False for reranker.")
            update_prometheus_config(rollout_config.prometheus, self.reranker_server_addresses)
    
    def _init_agent_loop_workers(self):
        """Initialize agent loop workers with dual agent support."""
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"co_search_agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.server_handles,
                    self.reranker_server_handles,
                    self.reward_router_address,
                )
            )
    
    def generate_sequences(self, prompts: DataProto) -> DataProto | tuple[DataProto, DataProto]:
        """Generate sequences with support for different training/validation modes.
        
        Args:
            prompts: Input batch with meta_info containing 'validate' flag.
            
        Returns:
            - If validate=True or reranker treated as tool: Single DataProto (main agent only)
            - Otherwise: Tuple of (main_agent_batch, reranker_agent_batch) for training
        """
        is_validate = prompts.meta_info.get("validate", False)
        global_steps = prompts.meta_info.get("global_steps", -1)
        val_start_step = self.config.trainer.get("reranker_sampling_val_start_step", -1)
        
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.wake_up()
        
        # Run workers
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get([
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
        ])
        
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.sleep()
        
        # Handle different return types based on validate flag or reranker-as-tool mode
        if is_validate:
            # Main-only mode: workers return DataProto directly
            main_outputs = outputs
            main_output = DataProto.concat(main_outputs)
            
            # Calculate performance metrics
            main_metrics = [output.meta_info.pop("metrics") for output in main_outputs]
            main_timing, main_aggregated_metrics = self._performance_metrics(main_metrics, main_output)
            main_output.meta_info = {
                "timing": main_timing, 
                "aggregated_metrics": main_aggregated_metrics, 
                **main_outputs[0].meta_info
            }
            
            return main_output
        else:
            # Training mode: workers return (main_batch, reranker_batch) tuples
            main_outputs = [output[0] for output in outputs]
            reranker_outputs = [output[1] for output in outputs]
            
            # Extract and aggregate reranker rollout metrics from all workers
            worker_reranker_metrics_list = [output.meta_info.pop("worker_reranker_metrics", {}) for output in reranker_outputs]
            aggregated_reranker_rollout_metrics = self._aggregate_reranker_metrics(worker_reranker_metrics_list)
            
            main_output = DataProto.concat(main_outputs)
            reranker_output = DataProto.concat(reranker_outputs)
            
            # Calculate performance metrics for main agent
            main_metrics = [output.meta_info.pop("metrics") for output in main_outputs]
            main_timing, main_aggregated_metrics = self._performance_metrics(main_metrics, main_output)
            main_output.meta_info = {
                "timing": main_timing, 
                "aggregated_metrics": main_aggregated_metrics, 
                **main_outputs[0].meta_info
            }
            
            # Calculate performance metrics for reranker
            reranker_metrics = [output.meta_info.pop("metrics") for output in reranker_outputs]
            reranker_timing, reranker_aggregated_metrics = self._performance_metrics(reranker_metrics, reranker_output)
            reranker_output.meta_info = {
                "timing": reranker_timing, 
                "aggregated_metrics": reranker_aggregated_metrics,
                "reranker_rollout_metrics": aggregated_reranker_rollout_metrics,  # Add aggregated metrics here
                **reranker_outputs[0].meta_info
            }
            
            return main_output, reranker_output
    
    def _aggregate_reranker_metrics(self, metrics_list: list[dict]) -> dict:
        """Aggregate reranker rollout metrics from multiple workers.
        
        合并规则:
        - avg_* : 加权平均 (按对应的 total/num)
        - max_* : 取最大值
        - min_* : 取最小值  
        - total_* : 求和
        - num_* : 求和
        - *_ratio : 重新计算 (基于求和后的分子/分母)
        - target_size : 取平均 (理论上相同)
        - fallback/triggered : 求和 (表示触发次数)
        """
        
        if not metrics_list or all(not m for m in metrics_list):
            return {}
        
        non_empty = [m for m in metrics_list if m]
        if not non_empty:
            return {}
        
        agg = {}
        all_keys = set(k for m in non_empty for k in m.keys())
        
        for key in all_keys:
            values = [m.get(key, 0.0) for m in non_empty]
            
            if "avg" in key:
                # 加权平均 - 需要找到对应的权重
                if "avg_final_score" in key or "avg_tool_score" in key or "avg_agent_score" in key:
                    # 按 total_outputs 加权
                    weights = [m.get(key.replace("avg", "total").replace("_score", "").replace("_final", "").replace("_tool", "").replace("_agent", "") + "/total_outputs", 1.0) for m in non_empty]
                    agg[key] = np.average(values, weights=weights)
                elif "avg_group_size" in key:
                    # 按 num_groups 加权
                    weights = [m.get(key.replace("avg_group_size", "num_groups"), 1.0) for m in non_empty]
                    agg[key] = np.average(values, weights=weights)
                else:
                    # 简单平均
                    agg[key] = np.mean(values)
            
            elif "max" in key:
                # 最大值
                agg[key] = np.max(values)
            
            elif "min" in key:
                # 最小值  
                agg[key] = np.min(values)
            
            elif "ratio" in key:
                # 比例需要重新计算
                if "duplicate/ratio" in key:
                    # num_duplicates / total_outputs
                    num_dup = sum(m.get("duplicate/num_duplicates", 0) for m in non_empty)
                    total = sum(m.get("duplicate/total_outputs", 0) for m in non_empty)
                    agg[key] = num_dup / total if total > 0 else 0.0
                elif "filter/filter_ratio" in key:
                    # total_filtered / total_groups  
                    filtered = sum(m.get("filter/total_filtered", 0) for m in non_empty)
                    groups = sum(m.get("pre_filter/num_groups", 0) for m in non_empty)
                    agg[key] = filtered / groups if groups > 0 else 0.0
                else:
                    # 未知 ratio - 简单平均
                    agg[key] = np.mean(values)
            
            elif key.startswith("total_") or key.startswith("num_") or "_total_" in key or "_num_" in key or "outputs" in key or "groups" in key or "filtered" in key or "duplicates" in key or "added" in key or "kept" in key or "triggered" in key:
                # 计数类指标 - 求和
                agg[key] = sum(values)
            
            elif "target_size" in key:
                # 常量 - 取平均 (应该相同)
                agg[key] = np.mean(values)
            
            else:
                # 默认求和
                agg[key] = sum(values)
        
        return agg


    def wake_up(self):
        """Wake up all rollout replica instances (main + reranker)."""
        super().wake_up()
        if hasattr(self, 'reranker_rollout_replicas'):
            self._run_all([replica.wake_up() for replica in self.reranker_rollout_replicas])
    
    def sleep(self):
        """Sleep all rollout replica instances (main + reranker)."""
        super().sleep()
        if hasattr(self, 'reranker_rollout_replicas'):
            self._run_all([replica.sleep() for replica in self.reranker_rollout_replicas])

