# Reranker Agent Setup Guide

## 概述

Reranker Agent 用于对检索结果进行重排序，提高文档相关性。它与 Search-R1 Agent 配合使用。

## 工作流程

```
Search-R1 Agent (step t)
    Messages: [user, assistant, tool, assistant, tool]
                                              ↓
                               Extract sub-query from last assistant
                                              ↓
Reranker Agent
    1. Call Dense Retrieval Tool → top-N documents
    2. Generate reranked document IDs → top-M documents
    3. Replace tool response → [user, assistant, tool, assistant, tool']
    4. (Optional) Call Reward Loop:
       - Run Search-R1 with updated messages → final answer
       - Compare with golden answer → reward
```

## 组件说明

### 1. Dense Retrieval Tool

调用密集检索 API 获取候选文档。

**配置文件**: `dense_retrieval_tool_config.yaml`
```yaml
tools:
  - class_name: verl.tools.dense_retrieval_tool.DenseRetrievalTool
    config:
      retrieval_service_url: http://127.0.0.1:8000/retrieve
      timeout: 30
      default_top_n: 200
      type: native
```

**API 格式**:
```python
# Request
POST http://127.0.0.1:8000/retrieve
{
    "query": "your search query",
    "top_k": 200
}

# Response
{
    "documents": [
        {
            "id": "doc_123",
            "content": "document text...",
            "score": 0.95
        },
        ...
    ]
}
```

### 2. Reranker Agent Loop

接收 sub-query 和文档，生成重排序结果。

**初始化**:
```python
from verl.experimental.agent_loop.reranker_agent_loop import RerankerAgentLoop
from verl.experimental.reward.reward_loop.reranker_reward_loop import RerankerRewardLoop

# Create reward loop (optional)
reward_loop = RerankerRewardLoop(
    config=config,
    tokenizer=tokenizer,
    search_r1_agent_loop=search_r1_agent_loop,
    n_rollouts=3,  # Run search-R1 3 times for each reranking
    reward_aggregation="mean"  # "mean", "max", or "median"
)

# Create reranker agent loop
reranker_agent_loop = RerankerAgentLoop(
    trainer_config=config,
    server_manager=reranker_server_manager,
    tokenizer=tokenizer,
    processor=None,
    reward_loop=reward_loop  # Optional
)
```

**运行**:
```python
# Prepare input messages (from Search-R1 at step t)
messages = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Let me search for information..."},
    {"role": "tool", "content": "Search results..."},
    {"role": "assistant", "content": "I need more specific information about deep learning"},
    {"role": "tool", "content": "More results..."}
]

# Run reranker agent
output = await reranker_agent_loop.run(
    sampling_params={
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9
    },
    raw_prompt=messages,
    multi_modal_data={},
    tools_kwargs={},
    # For reward loop
    original_messages=messages,  # Original messages for reward computation
    golden_answer="Machine learning is..."  # Expected answer
)

# Output contains:
# - output.extra_fields["reranked_document_ids"]: List of reranked doc IDs
# - output.extra_fields["reranking_reward"]: Reward from reward loop (if enabled)
# - output.metrics["num_retrieved_docs"]: Number of retrieved docs
# - output.metrics["num_reranked_docs"]: Number of reranked docs
```

### 3. Reranker Reward Loop

评估重排序质量的 reward loop。

**工作原理**:
1. 用 reranked documents 替换原始 tool response
2. 调用 Search-R1 Agent 生成 final answer
3. 与 golden answer 比较计算 reward（F1 score）
4. 可选：运行多次取平均/最大/中位数

**Reward 计算方式**:
- Exact Match: 完全匹配 = 1.0，否则 = 0.0
- Token F1: 基于 token overlap 的 F1 score
- (TODO) Semantic Similarity: 使用 embeddings
- (TODO) LLM-as-Judge: 使用 LLM 评估质量

## Reranker Output Format

Reranker Agent 应该输出重排序后的文档 ID 列表。支持的格式：

### Format 1: JSON
```json
{
    "ranked_ids": ["doc_5", "doc_1", "doc_12", "doc_8", "doc_3"]
}
```

### Format 2: Comma-separated
```
doc_5, doc_1, doc_12, doc_8, doc_3
```

### Format 3: Numbered list
```
1. doc_5
2. doc_1
3. doc_12
4. doc_8
5. doc_3
```

## 训练配置示例

```yaml
# dual_ppo_trainer.yaml
actor_rollout_ref:
  rollout:
    # Main agent (Search-R1)
    name: search_r1_agent
    multi_turn:
      tool_config_path: config/tool_config/retrieval_reranker_tool_config.yaml
      format: hermes
      max_assistant_turns: 5
      max_parallel_calls: 1
      max_tool_response_length: 4096
    
    # Reranker agent
    reranker:
      name: reranker_agent
      multi_turn:
        tool_config_path: config/tool_config/dense_retrieval_tool_config.yaml
        format: gpt-oss
        max_assistant_turns: 2
        max_parallel_calls: 1
        max_tool_response_length: 8192
      
      # Reward loop configuration
      reward_loop:
        enabled: true
        n_rollouts: 3
        reward_aggregation: mean  # "mean", "max", "median"
```

## 实现步骤

### Step 1: 启动 Dense Retrieval Service
```bash
python your_retrieval_service.py --port 8000
```

### Step 2: 自定义 Sub-Query 提取

在 `reranker_agent_loop.py` 中修改 `_extract_sub_query_from_messages()`:
```python
def _extract_sub_query_from_messages(self, messages: list[dict]) -> str:
    # 根据你的 assistant message 格式提取 query
    # 例如：从 tool call 中提取
    # 或者：从 <think> 标签中提取
    # 或者：使用正则表达式
    pass
```

### Step 3: 自定义 Reranked Document IDs 解析

在 `reranker_agent_loop.py` 中修改 `_parse_reranked_document_ids()`:
```python
def _parse_reranked_document_ids(self, response_text: str) -> list[str]:
    # 根据你的 reranker 输出格式解析
    # 支持 JSON、CSV、numbered list 等
    pass
```

### Step 4: 训练 Reranker

```bash
python -m verl.trainer.main_ppo \
    config=dual_ppo_trainer.yaml \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8
```

## 与 Search-R1 的集成

在 Search-R1 Agent 中使用 reranker：

```python
# In retrieval_reranker_tool.py

async def _call_reranker(
    self, 
    reranker_manager, 
    tokenizer,
    request_id: str,
    query: str, 
    documents: list[dict],
    top_m: int,
    sampling_params: dict[str, Any] = None
) -> list[str]:
    """Call reranker agent to rerank documents."""
    
    # Build reranker input messages
    # Format: [user, assistant, tool, assistant, tool]
    # The last assistant message contains the sub-query
    messages = [
        {"role": "user", "content": "Rerank the following documents"},
        {"role": "assistant", "content": f"Query: {query}"},
        {"role": "tool", "content": json.dumps({"documents": documents})},
        {"role": "assistant", "content": "I will rerank these documents"},
        {"role": "tool", "content": ""}  # Placeholder
    ]
    
    # Call reranker agent loop
    reranker_output = await reranker_agent_loop.run(
        sampling_params=sampling_params,
        raw_prompt=messages,
        multi_modal_data={},
        tools_kwargs={}
    )
    
    # Extract reranked document IDs
    reranked_ids = reranker_output.extra_fields.get("reranked_document_ids", [])
    return reranked_ids[:top_m]
```

## Metrics 说明

| Metric | 说明 |
|--------|------|
| `num_retrieved_docs` | Dense retrieval 返回的文档数 |
| `num_reranked_docs` | Reranker 输出的文档数 |
| `reranking_reward` | Reward loop 计算的奖励（如果启用）|
| `tool_calls/mean` | 平均 tool 调用次数 |

## 调试技巧

### 1. 检查 Sub-Query 提取
```python
logger.info(f"Extracted sub-query: {sub_query}")
```

### 2. 检查 Retrieved Documents
```python
logger.info(f"Retrieved {len(retrieved_documents)} documents")
logger.debug(f"First doc: {retrieved_documents[0]}")
```

### 3. 检查 Reranked IDs
```python
logger.info(f"Reranked document IDs: {reranked_document_ids}")
```

### 4. 检查 Reward
```python
logger.info(f"Reranking reward: {reranking_reward:.4f}")
```

## 常见问题

### Q: Sub-query 提取不正确？
A: 自定义 `_extract_sub_query_from_messages()` 方法，根据你的消息格式调整。

### Q: Reranked IDs 解析失败？
A: 检查 reranker 输出格式，调整 `_parse_reranked_document_ids()` 方法。

### Q: Reward 总是 0？
A: 检查 golden answer 是否正确，调整 reward 计算逻辑（exact match vs F1 vs semantic similarity）。

### Q: Dense retrieval API 连接失败？
A: 检查 `retrieval_service_url` 配置，确保 API 服务运行中。

## 下一步

1. 实现 `_extract_sub_query_from_messages()` - 根据你的消息格式
2. 实现 `_parse_reranked_document_ids()` - 根据你的 reranker 输出
3. 自定义 reward 计算逻辑（如果需要）
4. 启动 dense retrieval service
5. 开始训练！
