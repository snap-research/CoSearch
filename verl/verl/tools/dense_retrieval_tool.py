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

import json
import logging
import os
from typing import Any

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from .utils.search import call_search_api, format_tool_response

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DenseRetrievalTool(BaseTool):
    """Tool that calls dense retrieval API to get top-N documents.
    
    Used by Reranker Agent to retrieve candidate documents before reranking.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        # Set default tool schema if not provided
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema(
                type="function",
                function={
                    "name": "dense_retrieval",
                    "description": "Retrieve top-N relevant documents using dense retrieval API.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to retrieve documents."
                            },
                            "top_n": {
                                "type": "integer",
                                "description": "Number of documents to retrieve (default: 100).",
                                "default": 100
                            }
                        },
                        "required": ["query"]
                    }
                }
            )
        
        super().__init__(config, tool_schema)
        
        self.retrieval_url = config.get("retrieval_service_url")
        self.timeout = config.get("timeout", 30)
        self.default_top_n = config.get("default_top_n", 200)
        
        if not self.retrieval_url:
            raise ValueError("retrieval_service_url must be provided in config")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        """Execute dense retrieval.
        
        Args:
            instance_id: Tool instance ID
            parameters: Dict containing:
                - query: Search query string
                - top_n: Number of docs to retrieve (optional)
        
        Returns:
            Tuple of (ToolResponse, reward, metrics)
        """
        query = parameters.get("query")
        top_n = kwargs.get("top_n", self.default_top_n)
        
        if not query:
            return ToolResponse(text="Error: No query provided"), 0.0, {}
        
        metrics = {}
        
        try:
            # Call search API using utility function
            result = await call_search_api(
                query=query,
                search_api_url=self.retrieval_url,
                top_k=top_n,
                timeout=self.timeout
            )
            
            if result["status"] == "error":
                logger.error(f"Dense retrieval error: {result['error']}")
                return ToolResponse(text=f"Retrieval error: {result['error']}"), 0.0, metrics
            
            documents = result["documents"]
            metrics["num_retrieved_docs"] = len(documents)
            tool_response = format_tool_response(documents)
            
            # Format as JSON for reranker agent to parse
            response_text = json.dumps({
                "query": query,
                "response": tool_response,
                "documents": documents,
                "count": len(documents)
            }, ensure_ascii=False)
            
            return ToolResponse(text=response_text), 0.0, metrics
            
        except Exception as e:
            logger.error(f"Dense retrieval error: {e}")
            return ToolResponse(text=f"Retrieval error: {str(e)}"), 0.0, metrics

    async def calc_reward(self, instance_id: str, final_output: dict[str, Any], **kwargs):
        """Calculate reward for dense retrieval (optional, usually 0)."""
        return 0.0
