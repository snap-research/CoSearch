# SearchR1Agent Setup Guide

## 概述

`SearchR1AgentLoop` 是一个专门用于搜索任务的 agent loop，它使用 `RetrievalWithRerankerTool` 来执行：
1. Dense Retrieval（密集检索）
2. Reranker 重排序
3. 返回最相关的文档

## 架构

```
SearchR1AgentLoop
    ↓
RetrievalWithRerankerTool
    ├─ 调用 Dense Retrieval API (top-N)
    ├─ 调用 Reranker Agent (rerank)
    └─ 返回 Top-M Documents
```

## 配置步骤

### 1. 准备 Tool 配置文件

创建或使用 `retrieval_reranker_tool_config.yaml`：

```yaml
tools:
  - class_name: verl.tools.retrieval_reranker_tool.RetrievalWithRerankerTool
    config:
      retrieval_service_url: http://127.0.0.1:8000/retrieve
      timeout: 30
      default_top_n: 200
      default_top_m: 5
      sampling_params:
        temperature: 0.0
        max_tokens: 8192
        top_p: 1.0
      type: native
```

### 2. 在 Trainer 配置中引用

在你的训练配置文件中（如 `dual_ppo_trainer.yaml`）：

```yaml
actor_rollout_ref:
  rollout:
    name: agent_loop  # 使用 agent_loop rollout
    
    multi_turn:
      # 指向 tool 配置文件
      tool_config_path: examples/sglang_multiturn/config/tool_config/retrieval_reranker_tool_config.yaml
      
      # Tool parser 格式（根据你的 tokenizer）
      format: hermes  # 或 gpt-oss
      
      # 其他参数
      max_assistant_turns: 5
      max_parallel_calls: 1
      max_tool_response_length: 4096
      
      # Agent loop 配置路径（如果有自定义）
      agent_loop_config_path: null
```

### 3. 实现 Tool 的抽象方法

在 `retrieval_reranker_tool.py` 中实现这些方法：

#### `_build_reranker_prompt()`
```python
def _build_reranker_prompt(self, query: str, documents: list[dict]) -> str:
    """构建 reranker prompt"""
    prompt_lines = [
        f"Query: {query}",
        "",
        "Please rerank the following documents by relevance:",
        ""
    ]
    
    for i, doc in enumerate(documents, 1):
        doc_id = doc.get("id", f"doc_{i}")
        content = doc.get("content", "")[:500]  # 截断
        prompt_lines.append(f"{i}. [ID: {doc_id}]")
        prompt_lines.append(f"   {content}")
        prompt_lines.append("")
    
    prompt_lines.append("Output format: List document IDs in order, one per line.")
    return "\n".join(prompt_lines)
```

#### `_parse_reranker_output()`
```python
def _parse_reranker_output(self, reranked_text: str, documents: list[dict], top_m: int) -> list[str]:
    """解析 reranker 输出"""
    docids = []
    all_doc_ids = {doc.get("id") for doc in documents}
    
    # 从输出中提取 doc IDs
    for line in reranked_text.split("\n"):
        for doc_id in all_doc_ids:
            if doc_id in line and doc_id not in docids:
                docids.append(doc_id)
                if len(docids) >= top_m:
                    break
        if len(docids) >= top_m:
            break
    
    # Fallback
    if not docids:
        docids = [doc.get("id") for doc in documents[:top_m]]
    
    return docids[:top_m]
```

#### `_get_docs_by_ids()`
```python
def _get_docs_by_ids(self, documents: list[dict], doc_ids: list[str]) -> list[dict]:
    """根据 ID 获取文档"""
    doc_map = {doc.get("id"): doc for doc in documents}
    return [doc_map[doc_id] for doc_id in doc_ids if doc_id in doc_map]
```

#### `_format_documents()`
```python
def _format_documents(self, documents: list[dict]) -> str:
    """格式化文档为 tool response"""
    if not documents:
        return "No relevant documents found."
    
    lines = ["Search Results:", ""]
    for i, doc in enumerate(documents, 1):
        lines.append(f"{i}. {doc.get('content', '')[:500]}")
        lines.append("")
    
    return "\n".join(lines)
```

### 4. 启动 Retrieval API 服务

确保你的 dense retrieval API 运行在配置的 URL：

```bash
# 启动 retrieval service
python your_retrieval_service.py --port 8000
```

API 应该接受这样的请求：
```json
POST /retrieve
{
    "query": "machine learning",
    "top_k": 200
}
```

返回格式：
```json
{
    "documents": [
        {
            "id": "doc_123",
            "content": "...",
            "score": 0.95
        },
        ...
    ]
}
```

### 5. 运行训练

```bash
python -m verl.trainer.main_ppo \
    config=your_dual_ppo_trainer.yaml \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8
```

## Agent 使用示例

训练时，agent 会自动调用 tool：

```
User: "What is deep learning?"

SearchR1Agent (Generating):
"I need to search for information about deep learning."
<tool_call>search_with_rerank(query="deep learning", top_m=5)</tool_call>

System (Processing Tools):
- Calls retrieval API → gets 200 docs
- Calls reranker agent → reranks to top 5
- Returns top 5 docs to agent

SearchR1Agent (Generating):
"Based on the search results, deep learning is..."
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `retrieval_service_url` | Dense retrieval API 地址 | - |
| `default_top_n` | 检索多少个候选文档 | 200 |
| `default_top_m` | 重排后返回多少个文档 | 5 |
| `timeout` | API 超时时间（秒） | 30 |
| `sampling_params` | Reranker 采样参数 | 见配置文件 |

## 调试

### 检查 Tool 是否正确加载

```python
# 在 init_class 中会打印
print(f"Initialized tools: {cls.tools}")
# 应该看到: {'search_with_rerank': <RetrievalWithRerankerTool>}
```

### 检查 Reranker 是否被调用

在 tool 的 `execute()` 方法中添加日志：
```python
logger.info(f"Calling reranker with {len(documents)} documents")
```

### 查看 Metrics

在训练日志中会看到：
```
agent_loop/tool_calls/mean: 1.5
num_retrieved_docs: 200
num_reranked_docs: 5
```

## 常见问题

### Q: Reranker 没有被调用？
A: 检查 `reranker_server_manager` 是否正确传递：
```python
# 在 dual_agent_loop.py 的 _run_agent_loop 中
agent_loop = hydra.utils.instantiate(
    ...,
    reranker_server_manager=self.reranker_server_manager,  # 确保这行存在
)
```

### Q: Tool 找不到？
A: 检查 `tool_config_path` 路径是否正确，以及 tool class 是否在 Python path 中。

### Q: Retrieval API 连接失败？
A: 检查 URL 和端口，确保 API 服务正在运行。

## 下一步

1. 根据你的 retrieval API 格式调整 `_call_retrieval_api()`
2. 根据你的 reranker 输出格式调整 `_parse_reranker_output()`
3. 自定义 `_format_documents()` 来适配你的 agent 需求
4. 开始训练！
