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

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
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
        initial_query: Optional[str] = None,
        reranker_output_list: Optional[list[AgentLoopOutput]] = None,
        answers: Optional[list[str]] = None
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}
        self.initial_query = initial_query
        self.reranker_output_list = reranker_output_list or []
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

        # Buffer for LLM-as-Judge: each entry awaits next_generation to complete
        # Each dict: {"initial_query", "sub_query", "top_5_documents", "reranker_output_index"}
        self.pending_judge_inputs: list[dict[str, Any]] = []
        # Completed judge inputs: pending entries that have been paired with next_generation_text
        # These are ready to be sent to the judge in a batch after the agent loop terminates.
        self.completed_judge_inputs: list[dict[str, Any]] = []


@register("search_r1_reranker_reward_agent")
class SearchR1RerankerRewardAgentLoop(AgentLoopBase):
    def __init__(self, trainer_config, server_manager, tokenizer, processor, 
                 reranker_server_manager=None, reranker_tokenizer=None, 
                 reward_server_manager=None, reward_tokenizer=None, 
                 reward_http_client=None, judge_semaphore=None, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        # 保存 reranker server manager 用于调用 reranker agent
        self.reranker_server_manager = reranker_server_manager
        self.reranker_tokenizer = reranker_tokenizer if reranker_tokenizer else tokenizer
        self.reward_server_manager = reward_server_manager
        self.reward_tokenizer = reward_tokenizer 
        
        # HTTP client for external reward judge server (3-node architecture)
        self.reward_http_client = reward_http_client

        # Read LLM-as-Judge config from trainer_config (with safe defaults)
        judge_config = {}
        if hasattr(trainer_config, 'config'):
            judge_config = trainer_config.config.get("reward_judge_model", {})
        elif hasattr(trainer_config, 'get'):
            judge_config = trainer_config.get("reward_judge_model", {})
        self.n_judge_samples = judge_config.get("n_judge_samples", 4)
        self.max_judge_prompt_length = judge_config.get("max_judge_prompt_length", 16384)
        self.judge_sampling_params = dict(judge_config.get("judge_sampling_params", {
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.95,
        }))
        
        # Check reward judge mode: "ray" (internal vLLM) or "http_server" (external)
        self.reward_judge_mode = judge_config.get("mode", "ray")
        
        # Use the shared Worker-level semaphore if provided; otherwise create a local one (fallback).
        # The Worker-level semaphore limits concurrency ACROSS all AgentLoop instances in the same worker,
        # which is what actually prevents overwhelming the judge server.
        if judge_semaphore is not None:
            self._judge_semaphore = judge_semaphore
            logger.info("Using shared Worker-level judge semaphore")
        else:
            raise ValueError("A shared judge_semaphore must be provided to SearchR1RerankerRewardAgentLoop to limit concurrency to the reward judge server")
        
        logger.info(f"LLM-as-Judge config: mode={self.reward_judge_mode}, "
                    f"n_judge_samples={self.n_judge_samples}, "
                    f"max_judge_prompt_length={self.max_judge_prompt_length}, "
                    f"judge_sampling_params={self.judge_sampling_params}")

        if reranker_server_manager is not None:
            logger.info("Reranker server manager is provided for SearchR1RerankerRewardAgentLoop")
            logger.info("Warning: reranker tokenizer is the same as main tokenizer" 
                    if reranker_tokenizer is None else "Using provided reranker tokenizer")
        else:
            logger.info("No reranker server manager provided; reranker functionality will be disabled")
        
        # Log reward judge backend
        if self.reward_http_client is not None:
            logger.info(f"Using HTTP client for reward judge (3-node architecture)")
        elif self.reward_server_manager is not None:
            logger.info(f"Using Ray-managed reward server (2-node architecture)")

    @property
    def _has_judge_backend(self) -> bool:
        """Whether any LLM judge backend (Ray or HTTP) is available."""
        return self.reward_server_manager is not None or self.reward_http_client is not None

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

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})
        initial_query = kwargs["extra_info"]["question"]
        answers = list(kwargs["reward_model"]["ground_truth"]["target"])
        
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            initial_query=initial_query,
            answers=answers,
            reranker_output_list=[],
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=None,
            interaction_kwargs={},
        )
        agent_data.extra_fields["uid"] = kwargs["uid"] # for GRPO

        # State machine loop
        worker_id = os.getpid()
        state = AgentState.PENDING
        iteration = 0
        while state != AgentState.TERMINATED:
            iteration += 1
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data, sampling_params)
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
            extra_fields=agent_data.extra_fields,
        )

        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        output.extra_fields.update({"json_correct": agent_data.json_correct, "one_tool_call_per_assistant": agent_data.one_tool_call_per_assistant})

        # Handle remaining pending judges that never got a next generation
        # (e.g., agent terminated right after tool call due to max length)
        # These entries have no next_generation_text, so we set judge_score = None
        for pending in agent_data.pending_judge_inputs:
            idx = pending["reranker_output_index"]
            if 0 <= idx < len(agent_data.reranker_output_list):
                agent_data.reranker_output_list[idx].extra_fields.setdefault("llm_judge_score", None)
        agent_data.pending_judge_inputs.clear()

        # Batch-process ALL deferred judge calls after the agent loop terminates.
        # This avoids blocking the main generation loop and allows maximum GPU utilization.
        if agent_data.completed_judge_inputs and self._has_judge_backend:
            await self._process_completed_judges_batch(agent_data)

        return output, agent_data.reranker_output_list

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

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Pair pending LLM-as-Judge entries with next_generation_text (deferred — no await here)
        # Judge calls will be batched after the agent loop terminates to avoid blocking generation.
        if agent_data.pending_judge_inputs and self._has_judge_backend:
            next_generation_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            for pending in agent_data.pending_judge_inputs:
                pending["next_generation_text"] = next_generation_text
                agent_data.completed_judge_inputs.append(pending)
            agent_data.pending_judge_inputs.clear()
            
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
            return AgentState.PROCESSING_TOOLS
        else:
            agent_data.one_tool_call_per_assistant = False
            logger.warning(f"[W{worker_id}-GEN] no valid tools (got {num_calls}), terminating")
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []

        # 注入 reranker_server_manager 到 tools_kwargs，让 tool 内部可以调用 reranker
        # Use deep copy to avoid modifying shared nested dictionaries
        tools_kwargs_with_reranker = copy.deepcopy(agent_data.tools_kwargs)
        
        for tool_name in self.tools.keys():
            if tool_name not in tools_kwargs_with_reranker:
                tools_kwargs_with_reranker[tool_name] = {}
            
            if "create_kwargs" not in tools_kwargs_with_reranker[tool_name]:
                tools_kwargs_with_reranker[tool_name]["create_kwargs"] = {}
            
            create_kwargs = tools_kwargs_with_reranker[tool_name]["create_kwargs"]
            create_kwargs["reranker_tokenizer"] = self.reranker_tokenizer
            create_kwargs["request_id"] = agent_data.request_id
            create_kwargs["reranker_server_manager"] = self.reranker_server_manager
            create_kwargs["initial_query"] = agent_data.initial_query
            create_kwargs["answers"] = agent_data.answers
        # In search-r1 current context, only one tool call is allowed
        assert len(agent_data.tool_calls) == 1, "SearchR1AgentLoop only supports one tool call at a time."

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, tools_kwargs_with_reranker, agent_data=agent_data))
            tool_call_names.append(tool_call.name)
   
        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)
        
        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, res in responses:
            message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

            if "Error when executing tool:" in message["content"]:
                agent_data.json_correct = False

            # Buffer pending LLM judge input if reranker succeeded with valid format
            # Skip judge when: reranker crashed, or tool_score < 0 (format penalty)
            reranker_ok = (not res.get("reranker_crashed", True) 
                           and res.get("tool_score", 0.0) >= 0)
            if reranker_ok and self._has_judge_backend:
                agent_data.pending_judge_inputs.append({
                    "initial_query": agent_data.initial_query,
                    "sub_query": res.get("sub_query", ""),
                    "top_5_documents": tool_response.text or "",
                    "reranker_output_index": len(agent_data.reranker_output_list) - 1,
                })

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
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData,
    ) -> tuple[AgentToolResponse, float, dict]:
        """Call tool and return tool response."""
        worker_id = os.getpid()
        tool, instance_id = None, None
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
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
        tool_execution_response.text = tool_response_text
        
        # reranker agent output as expected 
        if not res["reranker_crashed"]:
            rr_output = AgentLoopOutput(
                    prompt_ids=tool_execution_response.prompt_ids,
                    response_ids=tool_execution_response.response_ids,
                    response_mask=tool_execution_response.response_mask,
                    response_logprobs=tool_execution_response.response_logprobs,
                    num_turns=2,
                    metrics=AgentLoopMetrics()
                )
            rr_output.extra_fields.update({"main_uid": agent_data.extra_fields["uid"],
                                           "tool_score": res.get("tool_score", 0.0),
                                           "answer_in_docs": res.get("answer_in_docs", False),
                                           "reranker_crashed": res.get("reranker_crashed", False),
                                           "sub_query": res.get("sub_query", "")})
            agent_data.reranker_output_list.append(
                rr_output
            )

        return tool_execution_response, tool_reward, res

    # ==================== LLM-as-Judge Methods ====================
    async def _process_completed_judges_batch(self, agent_data: AgentData):
        """Batch-process ALL completed judge inputs after the agent loop terminates.
        
        Unlike _process_pending_judges (which was called inline during generation),
        this method fires ALL judge calls for ALL entries as concurrent asyncio tasks,
        maximizing throughput to the reward model replicas.
        
        This is the key performance optimization: by deferring all judge calls to after
        the agent loop terminates, we avoid blocking the main generation loop. All 
        entries across all tool turns are processed in a single parallel batch.
        """
        print("process completed judges batch")
        n_judge_samples = self.n_judge_samples
        worker_id = os.getpid()
        
        logger.info(f"[W{worker_id}-JUDGE-BATCH] Processing {len(agent_data.completed_judge_inputs)} "
                     f"judge entries × {n_judge_samples} samples = "
                     f"{len(agent_data.completed_judge_inputs) * n_judge_samples} total calls")

        # Create ALL tasks across ALL entries at once
        all_tasks = []
        task_to_entry = []  # Maps task index back to entry index
        
        for entry_idx, pending in enumerate(agent_data.completed_judge_inputs):
            shared_judge_request_id = uuid4().hex
            for _ in range(n_judge_samples):
                task = self._call_llm_judge(
                    initial_query=pending["initial_query"],
                    sub_query=pending["sub_query"],
                    top_5_documents=pending["top_5_documents"],
                    next_generation=pending["next_generation_text"],
                    answers=agent_data.answers,
                    request_id=shared_judge_request_id,
                )
                all_tasks.append(task)
                task_to_entry.append(entry_idx)

        # Fire ALL judge calls concurrently
        all_results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Group results back by entry via slicing (tasks are ordered by entry)
        assert len(all_results) == len(agent_data.completed_judge_inputs) * n_judge_samples, (
            f"Mismatch: {len(all_results)} results != "
            f"{len(agent_data.completed_judge_inputs)} entries × {n_judge_samples} samples"
        )
        for entry_idx, pending in enumerate(agent_data.completed_judge_inputs):
            start = entry_idx * n_judge_samples
            entry_results = []
            for i in range(n_judge_samples):
                assert task_to_entry[start + i] == entry_idx, (
                    f"task_to_entry[{start + i}]={task_to_entry[start + i]} != entry_idx={entry_idx}"
                )
                result = all_results[start + i]
                if isinstance(result, Exception):
                    logger.error(f"[W{worker_id}-JUDGE-BATCH] Exception for entry {entry_idx}: {result}")
                    entry_results.append((None, ""))
                else:
                    entry_results.append(result)

            valid_scores = [score for score, _ in entry_results if score is not None]
            raw_texts = [text for _, text in entry_results]

            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
            else:
                avg_score = None
                logger.warning(f"[W{worker_id}-JUDGE-BATCH] All {n_judge_samples} judge calls failed for entry {entry_idx}")

            idx = pending["reranker_output_index"]
            if 0 <= idx < len(agent_data.reranker_output_list):
                agent_data.reranker_output_list[idx].extra_fields["llm_judge_score"] = avg_score
                agent_data.reranker_output_list[idx].extra_fields["llm_judge_raw_scores"] = valid_scores
                agent_data.reranker_output_list[idx].extra_fields["llm_judge_raw_texts"] = raw_texts
                logger.info(f"[W{worker_id}-JUDGE-BATCH] reranker[{idx}] judge_score={avg_score} "
                            f"(from {len(valid_scores)}/{n_judge_samples} valid samples)")

        agent_data.completed_judge_inputs.clear()

    async def _call_llm_judge(
        self, initial_query: str, sub_query: str, top_5_documents: str, next_generation: str, 
        answers: list[str], 
        request_id: str = None,
    ) -> tuple[Optional[float], str]:
        """Call LLM judge server once and return (parsed_score, raw_text).
        
        Uses a semaphore to limit concurrent calls and prevent overwhelming the server.
        
        Supports two modes:
        - Ray mode: Uses self.reward_server_manager (internal vLLM)
        - HTTP mode: Uses self.reward_http_client (external vLLM server)
        
        Args:
            request_id: Shared request_id for sticky session.
        
        Returns:
            Tuple of (parsed_score_or_None, raw_judge_text).
        """
        async with self._judge_semaphore:
            return await self._call_llm_judge_inner(
                initial_query, sub_query, top_5_documents, next_generation, answers, request_id
            )

    async def _call_llm_judge_inner(
        self, initial_query: str, sub_query: str, top_5_documents: str, next_generation: str, 
        answers: list[str], 
        request_id: str = None,
    ) -> tuple[Optional[float], str]:
        """Inner implementation of LLM judge call (called within semaphore)."""
        prompt = self._build_judge_prompt(initial_query, sub_query, top_5_documents, next_generation, answers)

        # HTTP mode (3-node architecture with external server)
        if self.reward_http_client is not None:
            try:
                # Use chat completions for HTTP mode (simpler, no tokenization needed)
                judge_messages = [{"role": "user", "content": prompt}]
                output = await self.reward_http_client.generate_chat(
                    request_id=request_id or uuid4().hex,
                    messages=judge_messages,
                    sampling_params=self.judge_sampling_params,
                )
                judge_text = output.text
                return self._parse_judge_score(judge_text), judge_text
            except Exception as e:
                logger.warning(f"LLM judge HTTP call failed: {e}")
                return None, ""
        
        # Ray mode (2-node architecture with internal server)
        elif self.reward_server_manager is not None:
            judge_messages = [{"role": "user", "content": prompt}]
            judge_prompt_ids = self.reward_tokenizer.apply_chat_template(
                judge_messages,
                add_generation_prompt=True,
                tokenize=True
            )
            # Truncate if too long
            judge_prompt_ids = judge_prompt_ids[:self.max_judge_prompt_length]

            try:
                # Remove max_tokens to avoid conflict with vllm_async_server
                judge_params = {k: v for k, v in self.judge_sampling_params.items() if k != "max_tokens"}
                output = await self.reward_server_manager.generate(
                    request_id=request_id or uuid4().hex,
                    prompt_ids=judge_prompt_ids,
                    sampling_params=judge_params,
                )
                # Decode prompt + output together for full quality inspection
                full_ids = judge_prompt_ids + list(output.token_ids)
                judge_text = self.reward_tokenizer.decode(full_ids, skip_special_tokens=True)
                return self._parse_judge_score(judge_text), judge_text
            except Exception as e:
                logger.warning(f"LLM judge Ray call failed: {e}")
                return None, ""
        
        else:
            logger.warning("No reward judge backend available (neither HTTP client nor Ray server)")
            return None, ""

    def _build_judge_prompt(self, initial_query: str, sub_query: str, 
                            top_5_documents: str, next_generation: str,
                            answers: list[str]) -> str:
        """Build the prompt for the LLM judge.
        
        Rubric: 0-3 scale evaluating reranker's document selection quality.
        See LLM_JUDGE_RUBRICS_DESIGN.md for full design rationale.
        """
        answers_formatted = "\n".join(f"- {a}" for a in answers)
        
        return f"""You are an expert judge evaluating the quality of documents selected by a reranker in a multi-step search system.

An AI assistant is answering a question through iterative search. At each step, it issues a search query, a reranker selects the top-ranked documents, and the assistant reads them to continue reasoning. Your job is to evaluate: How useful are the selected documents for reaching the correct answer?

## Correct Answer(s)
{answers_formatted}

## Original Question
{initial_query}

## Search Query (this step)
{sub_query}

## Top-Ranked Documents Selected by the Reranker
{top_5_documents}

## Assistant's Next Response (after reading the documents)
{next_generation}

## Scoring Rubric (0-3)
3 = Answer-Bearing: At least one document contains the correct answer, a clear paraphrase, or sufficient evidence to directly infer the answer.
2 = Useful Stepping Stone: Documents provide relevant intermediate facts that advance reasoning toward the answer (e.g., a necessary entity, relationship, or date for multi-hop reasoning), but the answer itself is not directly extractable.
1 = Topically Related but Unhelpful: Documents are about the same topic but contain no facts that support or advance toward the answer.
0 = Irrelevant: Documents have no connection to the query or answer.

## Guidelines
- Prioritize document content over assistant response quality. Good documents with a poor assistant response still deserve high scores.
- For multi-hop questions, intermediate facts that provide a necessary link in the reasoning chain count as valuable (score 2).
- Check all listed answer variants when assessing document-answer overlap.

## Response Format
First, do your step-by-step reasoning inside <think> tags. Then, output your final score (a single integer from 0 to 3) inside <score> tags.

<think>Your reasoning here</think>
<score>YOUR_SCORE</score>"""

    def _parse_judge_score(self, judge_text: str) -> Optional[float]:
        """Parse judge output and normalize to [0, 1] range.
        
        Raw score 0-3 maps to:
            0 → 0.0, 1 → 0.333, 2 → 0.667, 3 → 1.0
        """
        match = re.search(r'<score>\s*(\d+)\s*</score>', judge_text)
        if match:
            raw = int(match.group(1))
            raw = max(0, min(3, raw))  # Clamp to [0, 3]
            return raw / 3.0
        logger.warning(f"Failed to parse judge score from: {judge_text[-200:]}")
        return None

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