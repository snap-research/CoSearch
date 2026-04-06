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
import json
import logging
import os
from typing import Any, List, Optional
import ray

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse, AgentToolResponse
from .utils.search import call_search_api, format_tool_response, format_tool_response_with_docid_map
from .utils.prompts import RERANK_PROMPT_WITH_INITIAL_QUERY
from .utils.validates import validate_rerank_output
from .utils.answer_match_reward import compute_average_hit_at_ks, has_answer_in_documents, compute_ndcg_at_m

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CoSearchTool(BaseTool):
    """Search tool for CoSearch dual-agent system.

    This tool is called by the main agent with name "search" and supports two modes:
    1. Dense retrieval only (faster, no reranking)
    2. Dense retrieval + Reranker agent (more accurate)

    The mode is controlled by the `use_reranker` config parameter and the
    availability of reranker_server_manager in create_kwargs.

    Workflow:
    1. Call dense retrieval API to get top-N documents
    2. If reranker is enabled: Call reranker agent to rerank documents
    3. Return top-M documents (after reranking or directly from retrieval)
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        # Set default tool schema if not provided
        self._instance_kwargs = {}
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema(
                type="function",
                function={
                    "name": "search",  # Must match SEARCH_R1_PROMPT
                    "description": "Search for relevant documents to answer the user's question.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant information."
                            }
                        },
                        "required": ["query"]
                    }
                }
            )
        
        super().__init__(config, tool_schema)
        
        self.retrieval_url = config.get("retrieval_service_url")
        self.timeout = config.get("timeout", 30)
        self.default_top_n = config.get("default_top_n", 50)
        self.default_top_m = config.get("default_top_m", 5)
        self.format_penalty = config.get("format_penalty", -0.2)
        
        # Trivial answer filtering
        self.trivial_answers = set(
            a.lower().strip() for a in config.get("trivial_answers", ["yes", "no", "true", "false"])
        )
        
        # Tool score metric: "hit" (Average Hit@k) or "ndcg" (NDCG@M)
        self.tool_score_metric = config.get("tool_score_metric", "hit")
        if self.tool_score_metric not in ("hit", "ndcg"):
            raise ValueError(f"tool_score_metric must be 'hit' or 'ndcg', got '{self.tool_score_metric}'")
        
        # Retry configuration
        self.max_retries = config.get("max_retries", 3)  # Number of retry attempts
        self.retry_delay = config.get("retry_delay", 1.0)  # Initial retry delay in seconds
        self.retry_backoff = config.get("retry_backoff", 2.0)  # Exponential backoff multiplier
        
        # Save top-N documents to metrics for alternating training Phase 2.
        # Default: False (no impact on existing training scenarios).
        # Only set to True in Phase 1 of alternating training.
        self.save_top_n_documents = config.get("save_top_n_documents", False)

        # Reranker mode control:
        # True = dense retrieval + reranker (default for CoSearchTool)
        # False = retrieval-only (no reranker called, safe with reranker_server_manager=None)
        self.use_reranker = config.get("use_reranker", True)
        # Default reranker sampling params: greedy decoding (temperature=0) for fixed reranker
        self.reranker_sampling_params = config.get("reranker_sampling_params", {
            "temperature": 0.0,
            "top_p": 1.0
        })

        # Local per-process rate limiting: max 16 concurrent requests per worker
        # With 8 workers, this gives 8*16=128 total concurrent requests
        max_concurrent_per_worker = config.get("max_concurrent_per_worker", 16)
        self._semaphore = asyncio.Semaphore(max_concurrent_per_worker)
        logger.info(f"SearchR1Tool initialized with per-worker concurrency={max_concurrent_per_worker}")
        
        if not self.retrieval_url:
            raise ValueError("retrieval_service_url must be provided in config")

    async def create(self, instance_id=None, create_kwargs=None, **kwargs):
        instance_id, response = await super().create(instance_id, **kwargs)
    
        self._instance_kwargs[instance_id] = create_kwargs or {}
        return instance_id, response

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        """Execute search workflow.
        
        Args:
            instance_id: Tool instance ID
            parameters: Dict containing:
                - query: Search query string (required by SEARCH_R1_PROMPT)
        
        Returns:
            Tuple of (ToolResponse, reward, metrics)
        """
        create_kwargs = self._instance_kwargs.get(instance_id, {})
        query = parameters.get("query")

        top_m = create_kwargs.get("top_m", self.default_top_m)
        top_n = create_kwargs.get("top_n", self.default_top_n)
        initial_query = create_kwargs.get("initial_query")
        answers = create_kwargs.get("answers", [])

        
        if not query:
            logger.error("No query provided to SearchR1Tool.execute: query is empty or None")
            return AgentToolResponse(text="Error: No query provided"), 0.0, {"reranker_crashed": True}
        
        metrics = {}
        
        # Step 1: Call dense retrieval API
        try:
            documents = await self._call_retrieval_api(query, top_n)
            metrics["num_retrieved_docs"] = len(documents)
            # Save top-N documents for Phase 2 reranker training (alternating training only)
            if self.save_top_n_documents:
                metrics["top_n_documents"] = documents
        except Exception as e:
            # Log once with all context at warning level (errors already logged in _call_retrieval_api)
            logger.error(f"Retrieval failed for query '{query[:50]}...': {e}")
            metrics["reranker_crashed"] = True
            return AgentToolResponse(text=f"Retrieval error: {str(e)}"), 0.0, metrics
        
        # Check if answers are trivial (e.g., "yes", "no") — meaningless for doc relevancy
        answers_are_trivial = bool(
            self.trivial_answers
            and answers
            and all(a.lower().strip() in self.trivial_answers for a in answers)
        )
        metrics["answers_are_trivial"] = answers_are_trivial
        
        # check answer in top-N documents
        if answers_are_trivial:
            answer_in_docs = False
        else:
            loop = asyncio.get_event_loop()
            answer_in_docs = await loop.run_in_executor(
                None,
                lambda: has_answer_in_documents(
                    answers=answers,
                    documents=documents
                )
            )
        metrics["answer_in_docs"] = answer_in_docs
        metrics["sub_query"] = query
        
        # Step 2: call reranker (if enabled) or use retrieval results directly
        rerank_result = {}  # Initialize for agent_output extraction later
        metrics["reranker_success"] = False
        metrics["reranker_fallback"] = False
        metrics["reranker_fallback_reason"] = None
        metrics["reranker_crashed"] = False

        if not self.use_reranker:
            # Retrieval-only mode: skip reranker, use top-M from dense retrieval
            final_documents = documents[:top_m]
            metrics["tool_score"] = 0.0
            logger.info(f"Retrieval-only mode: returning top-{top_m} (use_reranker=False)")
        else:
            reranker_manager = create_kwargs.get("reranker_server_manager")
            tokenizer = create_kwargs.get("reranker_tokenizer")
            request_id = create_kwargs.get("request_id")
            sampling_params = create_kwargs.get("reranker_sampling_params", self.reranker_sampling_params)

            if not reranker_manager or not tokenizer:
                raise ValueError(
                    "Reranker manager and tokenizer must be provided in create_kwargs "
                    "when use_reranker is True. Set use_reranker: false for retrieval-only."
                )
            try:
                rerank_result = await self._call_reranker(
                    reranker_manager=reranker_manager, 
                    tokenizer=tokenizer, 
                    request_id=request_id,
                    initial_query=initial_query,
                    query=query, 
                    documents=documents, 
                    top_m=top_m, 
                    sampling_params=sampling_params,
                    reranker_as_agent=True
                )
                
                if rerank_result["status_message"] != "Success.":
                    # Format validation failed - fallback with tracking
                    logger.warning(
                        f"Reranker output format invalid: {rerank_result['errors']}. "
                        f"Falling back to retrieval top-{top_m}. "
                        f"Raw output: {rerank_result.get('raw_output', 'N/A')[:200]}"
                    )
                    final_documents = documents[:top_m]
                    metrics["reranker_fallback"] = True
                    metrics["reranker_fallback_reason"] = "format_validation_error"
                    metrics["reranker_validation_errors"] = rerank_result["errors"]
                    metrics["num_reranked_docs"] = 0
                    metrics["tool_score"] = self.format_penalty
                else:
                    # Success path
                    final_documents = rerank_result["reranked_docs"]
                    metrics["reranker_success"] = True
                    metrics["num_reranked_docs"] = len(rerank_result["reranked_docs"])
                    
            except Exception as e:
                # API/execution error - fallback with tracking
                logger.error(
                    f"Reranker execution error: {type(e).__name__}: {e}. "
                    f"Falling back to retrieval top-{top_m}."
                )
                final_documents = documents[:top_m]
                metrics["reranker_fallback"] = True
                metrics["reranker_fallback_reason"] = "execution_error"
                metrics["reranker_error_type"] = type(e).__name__
                metrics["reranker_error_message"] = str(e)
                metrics["num_reranked_docs"] = 0
                metrics["reranker_crashed"] = True
                metrics["tool_score"] = -1.0

        # Format response
        response_text = format_tool_response(final_documents)
        
        # Calculate reward (optional)
        reward = 0.0
        hit_cutoffs = create_kwargs.get("hit_cutoffs", [1,3,5])
        top_m = create_kwargs.get("top_m", self.default_top_m)

        loop = asyncio.get_running_loop()
        
        if self.tool_score_metric == "ndcg":
            if metrics["reranker_success"]:
                assert len(final_documents) == top_m, f"Final documents length {len(final_documents)} does not match top_m {top_m} for NDCG computation"
                # rerank_result["reranked"] is 1-indexed → convert to 0-indexed
                reranked_indices = [i - 1 for i in rerank_result["reranked"]]
                reward, num_relevant = await loop.run_in_executor(
                    None,
                    lambda: compute_ndcg_at_m(
                        answers=answers,
                        all_documents=documents,        # top-N pool (50 docs) for qrel
                        ranked_indices=reranked_indices, # 0-indexed positions in reranker order
                        top_m=top_m
                    )
                )
                metrics["ndcg_at_m"] = reward
                metrics["num_relevant_in_pool"] = num_relevant
                # print("[In ndcg computation] reward (NDCG@M):", reward, "num_relevant_in_pool:", num_relevant)
        else:
            reward = await loop.run_in_executor(
                None,
                lambda: compute_average_hit_at_ks(
                    answers=answers,
                    documents=final_documents,
                    hit_cutoffs=hit_cutoffs
                )
            )
            metrics["average_hit_at_ks"] = reward
        
        # Only override tool_score if reranker succeeded (not format error or crash)
        if metrics["reranker_success"]:
            # If answer_in_docs is False (including trivial answers), tool_score is always 0
            if not answer_in_docs:
                metrics["tool_score"] = 0.0
            else:
                metrics["tool_score"] = reward
        if metrics["reranker_crashed"]:
            logger.error("Reranker crashed; but we set reranker as the agent.")
            return AgentToolResponse(text=response_text), reward, metrics

        agent_output = rerank_result.get("agent_output", None)
        if agent_output:
            return AgentToolResponse(
                text=response_text,
                prompt_ids=agent_output["prompt_ids"],
                response_ids=agent_output["response_ids"],
                response_mask=agent_output["response_mask"],
                response_logprobs=agent_output["response_logprobs"]
            ), reward, metrics
        else:
            return AgentToolResponse(text=response_text), reward, metrics

    async def _call_retrieval_api(self, query: str, top_n: int) -> list[dict]:
        """Call dense retrieval API to get top-N documents with retry logic.
        
        Args:
            query: Search query
            top_n: Number of documents to retrieve
        
        Returns:
            List of retrieved documents
        
        Raises:
            Exception: If all retry attempts fail
        """
        last_error = None
        retry_delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                # Acquire local semaphore slot (blocks if at capacity)
                async with self._semaphore:
                    result = await call_search_api(
                        query=query,
                        search_api_url=self.retrieval_url,
                        top_k=top_n,
                        semaphore=None,  # Not using nested semaphore
                        timeout=self.timeout
                    )
                    
                    if result["status"] == "error":
                        raise Exception(result["error"])
                    
                    # Success - log if we had retries
                    if attempt > 0:
                        logger.info(f"Retrieval succeeded on attempt {attempt + 1}/{self.max_retries}")
                    
                    return result["documents"]
                    
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Retrieval attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= self.retry_backoff  # Exponential backoff
                else:
                    logger.error(
                        f"Retrieval failed after {self.max_retries} attempts. Last error: {e}"
                    )
        
        # All retries exhausted
        raise Exception(f"Retrieval failed after {self.max_retries} attempts: {last_error}")
    
    async def _call_reranker(
        self, 
        reranker_manager, 
        tokenizer,
        request_id: str,
        initial_query: str, 
        query: str, 
        documents: list[dict],
        top_m: int,
        sampling_params: dict[str, Any],
        reranker_as_agent: bool = True
    ) -> dict:
        """Call reranker agent to rerank documents.
        
        Args:
            reranker_manager: AsyncLLMServerManager for reranker
            tokenizer: Tokenizer instance
            request_id: Request ID for tracking
            initial_query: Initial search query
            query: Original search query
            documents: List of retrieved documents
            top_m: Number of top documents to return
        
        Returns:
            Dict with rerank results
        """
        prompt_text, docid_map = self._build_reranker_prompt(
            initial_query=initial_query,
            query=query, 
            documents=documents, 
            top_m=top_m
        )
        
        # Tokenize
        reranker_messages = [{"role": "user", "content": prompt_text}]
        reranker_prompt_ids = tokenizer.apply_chat_template(
            reranker_messages,
            add_generation_prompt=True,
            tokenize=True
        )
        
        # Check and truncate prompt if too long (max 16384 tokens)
        max_prompt_length = 16384
        if len(reranker_prompt_ids) > max_prompt_length:
            logger.warning(
                f"Reranker prompt too long ({len(reranker_prompt_ids)} tokens). "
                f"Truncating to {max_prompt_length} tokens."
            )
        reranker_prompt_ids = reranker_prompt_ids[:max_prompt_length]
        
        reranker_output = await reranker_manager.generate(
            request_id=request_id,
            prompt_ids=reranker_prompt_ids,
            sampling_params=sampling_params,
        )

        # Parse reranker output to extract top-M doc IDs
        reranked_text = tokenizer.decode(reranker_output.token_ids, skip_special_tokens=True)
        rerank_result = validate_rerank_output(
            output=reranked_text,
            N=len(documents),
            M=top_m,
            docid_map=docid_map
        )

        response_mask = [1] * len(reranker_output.token_ids)
        response_length = 4096
        rerank_result["agent_output"] = {
            "prompt_ids": reranker_prompt_ids[:max_prompt_length],
            "response_ids": reranker_output.token_ids[:response_length],
            "response_mask": response_mask[:response_length],
            "response_logprobs": reranker_output.log_probs[:response_length] if reranker_output.log_probs else None,
        }

        return rerank_result

    def _build_reranker_prompt(self, initial_query: str, query: str, documents: list[dict], top_m: int) -> tuple[str, dict]:
        """Build prompt for reranker model.
        
        Args:
            initial_query: Initial search query
            query: Original search query
            documents: List of documents with id and content
            top_m: Number of top documents to return
        
        Returns:
            Tuple of (formatted prompt string, docid_map)
        """
        passage_block, docid_map = format_tool_response_with_docid_map(documents)
        prompt = RERANK_PROMPT_WITH_INITIAL_QUERY.format(
            N=len(documents),
            M=min(top_m, len(documents)),
            initial_query=initial_query,
            sub_query=query,
            passages_block=passage_block
        )
        return prompt, docid_map
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_kwargs:
            del self._instance_kwargs[instance_id]