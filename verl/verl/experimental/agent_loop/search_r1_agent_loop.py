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
import re
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import add_generation_prompt_for_gpt_oss, format_gpt_oss_tool_response_manually
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse, AgentToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
        answers: Optional[list[str]] = None
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}
        self.answers = answers
        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # Extra fields for dynamic addition
        self.extra_fields: dict[str, Any] = {}

        self.json_correct: bool = True 
        self.one_tool_call_per_assistant: bool = True

        # for dual agent (reranker) usage
        self.executed_tool_calls: list[FunctionCall] = []


@register("search_r1_agent")
class SearchR1AgentLoop(AgentLoopBase):
    def __init__(self, trainer_config, server_manager, tokenizer, processor, 
                 reranker_server_manager=None, reranker_tokenizer=None, 
                 track_messages=False,  **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        # 保存 reranker server manager 用于调用 reranker agent
        self.reranker_server_manager = reranker_server_manager
        self.reranker_tokenizer = reranker_tokenizer if reranker_tokenizer else tokenizer

        if reranker_server_manager is not None:
            logger.info("Reranker server manager is provided for SearchR1AgentLoop")
            logger.info("Warning: reranker tokenizer is the same as main tokenizer" 
                    if reranker_tokenizer is None else "Using provided reranker tokenizer")
        else:
            logger.info("No reranker server manager provided; reranker functionality will be disabled")

        # Read track_messages from config, only activate when training reranker (reranker as agent tool)
        self.track_messages = track_messages
        logger.info(f"track_messages is set to: {self.track_messages}")

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        print(f"\n[SEARCH R1 AGENT] Tool config path: {tool_config_path}")
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = None # we don't need tool schemas
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format
        print(f"[SEARCH R1 AGENT] Initialized tools: {list(cls.tools.keys())}")
        print(f"[SEARCH R1 AGENT] Tool instances: {cls.tools}\n")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

        # check tools are only for search 
        assert len(tool_list) == 1, "SearchR1AgentLoop only supports one tool for search."
        assert tool_list[0].name == "search", "The only supported tool is 'search'."
        assert cls.tool_schemas is None, "Tool schemas are not needed for SearchR1AgentLoop."

    def _infer_user_and_assistant_turns(self, messages: list[dict[str, Any]]) -> tuple[int, int]:
        """Infer the number of user and assistant turns from the message history."""
        if len(messages) == 1:
            return 0, 0
        else:
            user_turns = sum(1 for msg in messages if msg["role"] == "tool") - 1 
            assistant_turns = sum(1 for msg in messages if msg["role"] == "assistant")
        return user_turns, assistant_turns

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})
        answers = list(kwargs["reward_model"]["ground_truth"]["target"])
        
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=None,
            interaction_kwargs={},
            answers=answers,
        )

        # we may need to infer user and assistant turns from messages 
        # it is used for we treat reranker as an agent tool
        if self.track_messages:
            user_turns, assistant_turns = self._infer_user_and_assistant_turns(messages)
            agent_data.user_turns = user_turns
            agent_data.assistant_turns = assistant_turns

        # State machine loop
        worker_id = os.getpid()
        state = AgentState.PENDING
        iteration = 0
        while state != AgentState.TERMINATED:
            iteration += 1
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
                logger.debug(f"[W{worker_id}#{iteration}] PENDING→{state.value} | U{agent_data.user_turns}/A{agent_data.assistant_turns}")
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
                logger.debug(f"[W{worker_id}#{iteration}] GENERATING→{state.value} | U{agent_data.user_turns}/A{agent_data.assistant_turns}")
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data, sampling_params)
                logger.debug(f"[W{worker_id}#{iteration}] PROCESSING_TOOLS→{state.value} | U{agent_data.user_turns}/A{agent_data.assistant_turns}")
            else:
                logger.error(f"[W{worker_id}] Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        if self.track_messages:
            output.extra_fields["messages"] = agent_data.messages
            output.extra_fields["executed_tool_calls"] = agent_data.executed_tool_calls

        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        output.extra_fields.update({"json_correct": agent_data.json_correct, "one_tool_call_per_assistant": agent_data.one_tool_call_per_assistant})
        
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []
        worker_id = os.getpid()

        logger.debug(f"[W{worker_id}-GEN] start p_len={len(agent_data.prompt_ids)} mask_len={len(agent_data.response_mask)}")
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )
        logger.debug(f"[W{worker_id}-GEN] done out_len={len(output.token_ids)}")

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if self.track_messages:
            assistant_content = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids)
            )
            message = {"role": "assistant", "content": assistant_content}
            agent_data.messages.append(message)
            
        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # let's check whether we get the answer first, if so, terminate directly
        if self.detect_answer(agent_data.response_ids):
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        num_calls = len(agent_data.tool_calls) if agent_data.tool_calls else 0

        # Determine next state, we only allow 1 tool call per generation turn
        if agent_data.tool_calls and len(agent_data.tool_calls) == 1:
            logger.debug(f"[W{worker_id}-GEN] extracted 1 tool: {agent_data.tool_calls[0].name}")
            return AgentState.PROCESSING_TOOLS
        else:
            agent_data.one_tool_call_per_assistant = False
            logger.warning(f"[W{worker_id}-GEN] no valid tools (got {num_calls}), terminating")
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []

        # 注入 reranker_server_manager 到 tools_kwargs，让 tool 内部可以调用 reranker
        tools_kwargs_with_reranker = dict(agent_data.tools_kwargs)
        
        # 准备 create_kwargs for all tools
        for tool_name in self.tools.keys():
            if tool_name not in tools_kwargs_with_reranker:
                tools_kwargs_with_reranker[tool_name] = {}
            
            # 设置 create_kwargs
            if "create_kwargs" not in tools_kwargs_with_reranker[tool_name]:
                tools_kwargs_with_reranker[tool_name]["create_kwargs"] = {}
            
            # 注入必要的依赖
            create_kwargs = tools_kwargs_with_reranker[tool_name]["create_kwargs"]
            create_kwargs["reranker_tokenizer"] = self.reranker_tokenizer
            create_kwargs["request_id"] = agent_data.request_id
            create_kwargs["answers"] = agent_data.answers
            
            # 注入主模型的 sampling_params（用于 fallback 或其他用途）
            create_kwargs["sampling_params"] = sampling_params
            
            # 如果有 reranker，注入 reranker_server_manager 和 reranker_sampling_params
            if self.reranker_server_manager is not None:
                create_kwargs["reranker_server_manager"] = self.reranker_server_manager
                create_kwargs["reranker_sampling_params"] = self.tools[tool_name].config["reranker_sampling_params"]
        # In search-r1 current context, only one tool call is allowed
        assert len(agent_data.tool_calls) == 1, "SearchR1AgentLoop only supports one tool call at a time."

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, tools_kwargs_with_reranker))
            tool_call_names.append(tool_call.name)
        
        if self.track_messages:
            agent_data.executed_tool_calls.extend(agent_data.tool_calls[: self.max_parallel_calls])
   
        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)
        
        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses:
            message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

            if "Error when executing tool:" in message["content"]:
                agent_data.json_correct = False

        assert len(add_messages) == 1, "SearchR1AgentLoop only supports one tool call at a time."
        agent_data.messages.extend(add_messages)
        # Update prompt with tool responses
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Use only the new images from this turn for processing tool responses
            current_images = new_images_this_turn if new_images_this_turn else None  # Using local variable
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if self.tool_parser_name == "gpt-oss":
                logger.info("manually format tool responses for gpt-oss")
                # Format tool responses manually
                tool_response_texts = []
                for i, tool_msg in enumerate(add_messages):
                    actual_tool_name = tool_call_names[i]
                    formatted = format_gpt_oss_tool_response_manually(tool_msg["content"], actual_tool_name)
                    tool_response_texts.append(formatted)

                tool_response_text = add_generation_prompt_for_gpt_oss("".join(tool_response_texts))
                response_ids = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
                )
            else:
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
                )
                response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1

        return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        worker_id = os.getpid()
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)
          
        except Exception as e:
            logger.warning(f"[W{worker_id}-TOOL-{tool_name}] ERROR: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)
                logger.debug(f"[W{worker_id}-TOOL-{tool_name}] release() done")

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def detect_answer(self, response_ids: list[int]) -> bool:
        """Detect if the model has provided a final answer in its response.

        The function looks for the presence of an <answer>...</answer> tag
        in the decoded text of the response IDs.

        Example valid output:
        <reason>...</reason>
        <answer>...</answer>

        Args:
            response_ids: The list of token IDs generated by the model.
        Returns:
            True if a final answer is detected, False otherwise.
        """
        text = self.tokenizer.decode(response_ids)
        # Use strict regex to match <answer>...</answer>
        answer_regex = r"<answer>(.*?)</answer>"
        match = re.search(answer_regex, text, re.DOTALL)
        return match is not None
    
    async def call_tool_as_agent(
        self, tool_call: FunctionCall, answers: list[str],
    ) -> tuple[AgentToolResponse, float, dict]:
        """Call tool and return tool response."""
        worker_id = os.getpid()
        tool, instance_id = None, None

        # we build tools_kwargs by ourself  
        tools_kwargs = {tool_call.name: {}}
        tools_kwargs[tool_call.name]["create_kwargs"] = {}
        tools_kwargs[tool_call.name]["create_kwargs"]["reranker_tokenizer"] = self.reranker_tokenizer
        tools_kwargs[tool_call.name]["create_kwargs"]["request_id"] = uuid4().hex
        tools_kwargs[tool_call.name]["create_kwargs"]["reranker_server_manager"] = self.reranker_server_manager
        tools_kwargs[tool_call.name]["create_kwargs"]["reranker_sampling_params"] = self.tools[tool_call.name].config["reranker_agent_sampling_params"]
        tools_kwargs[tool_call.name]["create_kwargs"]["hits_cutoffs"] = self.tools[tool_call.name].config.get("hit_cutoffs", [1,3,5])
        
        # Per-call kwargs passed as parameters (not create_kwargs) to avoid concurrent overwrites
        per_call_kwargs = {
            "reranker_as_agent": True,
            "compute_tool_reward": True,
            "answers": answers
        }
        # print("call tool as agent here, answers =", answers)

        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            # Merge per-call kwargs into tool_args (passed as parameters to execute)
            tool_args.update(per_call_kwargs)
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)
          
        except Exception as e:
            logger.warning(f"[W{worker_id}-TOOL-{tool_name}] ERROR: {e}")
            return (
                AgentToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {"reranker_crashed": True},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)
                logger.debug(f"[W{worker_id}-TOOL-{tool_name}] release() done")

        if res["reranker_crashed"]:
            logger.error("Reranker crashed durning call tool when treat reranker as agent.")
            return (
                AgentToolResponse(
                    text="Error when executing tool: Reranker crashed during execution.",
                ),
                0.0,
                {"reranker_crashed": True},
            )

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create AgentToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text, 
                                "prompt_ids": tool_execution_response.prompt_ids,
                                "response_ids": tool_execution_response.response_ids,
                                "response_mask": tool_execution_response.response_mask,
                                "response_logprobs": tool_execution_response.response_logprobs
                                }

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        # print("call tool as agent here, res =", res)
        return AgentToolResponse(**tool_response_kwargs), tool_reward, res
