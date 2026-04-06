# Search-R1 Dual Agent Architecture

## Overview

这个架构实现了 Search-R1 系统的双 agent 训练，包括：
1. **Main Agent (Search-R1)**: 负责查询重写、工具调用、推理生成
2. **Reranker Agent**: 负责对检索文档进行重排序

## 核心组件

### 1. SearchR1AgentLoop (已存在)
**位置**: `search_r1_agent_loop.py`

**功能**:
- 实现完整的 Search-R1 agent 逻辑
- 支持工具调用（search tool）
- 已经内置 reranker_server_manager 支持
- 在 `_handle_processing_tools_state` 中调用 reranker

**关键参数**:
```python
def __init__(
    self,
    trainer_config,
    server_manager,           # Main agent server
    tokenizer,
    processor,
    reranker_server_manager=None,  # Reranker server (可选)
    reranker_tokenizer=None,       # Reranker tokenizer (可选)
    **kwargs
):
```

**使用场景**:
- 单独使用时：只用 main agent
- 传入 reranker_server_manager 时：自动使用 reranker 进行文档重排

---

### 2. SearchR1DualAgentLoopManager (新增)
**位置**: `agent_loop.py` (末尾)

**功能**:
- 管理两组 vLLM/SGLang server (main + reranker)
- 支持两种训练模式切换
- 使用 `ExtendedAgentLoopWorker` 作为 worker 类

**初始化**:
```python
manager = SearchR1DualAgentLoopManager(
    config=config,
    worker_group=main_agent_wg,        # Main agent worker group
    reranker_worker_group=reranker_wg, # Reranker worker group
    rm_wg=rm_wg                        # Reward model worker group (可选)
)
```

**核心方法**:
```python
# 主入口，根据 train_mode 切换逻辑
output = manager.generate_sequences(batch, train_mode="search_r1")
# 或
output = manager.generate_sequences(batch, train_mode="reranker")
```

---

### 3. ExtendedAgentLoopWorker (复用)
**位置**: `dual_agent_loop_extended.py`

**功能**:
- Worker 类，处理实际的 generation 逻辑
- 支持 counterfactual rollout (用于训练 reranker)
- 管理两组 server handles

**初始化**:
```python
worker = ExtendedAgentLoopWorker.remote(
    config,
    server_handles,           # Main agent server handles
    reranker_server_handles,  # Reranker server handles
    reward_router_address
)
```

**关键方法**:
- `generate_sequences()`: 标准生成（训练 main agent）
- `generate_sequences_counterfactual()`: Counterfactual rollout（训练 reranker）

---

## 两种训练模式

### Mode 1: Train Search-R1 (train_mode="search_r1")

**目标**: 训练主 agent，reranker 固定

**流程**:
1. Main agent: 可训练，temperature > 0
2. Reranker: 固定（deterministic），temperature = 0
3. 标准 agent loop 执行
4. Reward 基于最终答案的 EM/F1

**调用**:
```python
output = manager.generate_sequences(batch, train_mode="search_r1")
```

**对应方法**: `_generate_sequences_train_search_r1()`

---

### Mode 2: Train Reranker (train_mode="reranker")

**目标**: 训练 reranker，main agent 固定

**流程**:
1. 运行初始 rollout 获取完整轨迹
2. 识别所有工具调用位置作为分支点
3. 对每个分支点：
   - 采样 4 个不同的 reranker 响应（temperature > 0）
   - 从每个分支继续生成（main agent 固定）
4. 按分支点分组用于 GRPO
5. 基于最终答案计算 reward

**调用**:
```python
output = manager.generate_sequences(batch, train_mode="reranker")
```

**对应方法**: `_generate_sequences_train_reranker()`

**核心特性**:
- **Counterfactual Rollout**: 在同一轨迹的不同时间点尝试不同的 reranker 决策
- **GRPO Grouping**: 每个分支点的 4 个分支形成一个 GRPO group
- **Main Agent Fixed**: Main agent 使用确定性生成（temperature=0）

---

## Config 结构

```yaml
actor_rollout_ref:
  # Main agent configuration
  model:
    path: /path/to/search-r1-model
  rollout:
    mode: async
    tensor_model_parallel_size: 1
    data_parallel_size: 1
    # ... other rollout configs
    agent:
      agent_loop_config_path: /path/to/agent_loop_config.yaml
      num_workers: 4

reranker_rollout:
  # Reranker configuration (structure mirrors actor_rollout_ref)
  model:
    path: /path/to/reranker-model
  rollout:
    mode: async
    tensor_model_parallel_size: 1
    data_parallel_size: 1
    # ... other rollout configs
  trainable: false  # or true
  
  # Counterfactual rollout configs (only used when training reranker)
  num_branches: 4
  branch_temperature: 1.0
```

---

## 使用示例

### 在 DualAgentRayTrainer 中使用

```python
# dual_agent_ray_trainer.py

def init_workers(self):
    # ... 初始化 worker groups ...
    
    # Create agent loop manager
    from verl.experimental.agent_loop.agent_loop import SearchR1DualAgentLoopManager
    
    self.async_rollout_manager = SearchR1DualAgentLoopManager(
        config=self.config,
        worker_group=self.actor_rollout_wg,
        reranker_worker_group=self.reranker_actor_rollout_wg,
        rm_wg=self.rm_wg,
    )

def fit(self):
    for batch in dataloader:
        # Update training mode
        self._update_train_mode()
        
        # Generate with current training mode
        gen_batch_output = self.async_rollout_manager.generate_sequences(
            gen_batch_output,
            train_mode=self.current_train_mode  # "search_r1" or "reranker"
        )
        
        # ... rest of training loop ...
```

---

## 关键设计决策

### 1. 为什么不需要新的 AgentLoop 类？

**SearchR1AgentLoop 已经足够**:
- 它实现了 `AgentLoopBase` 接口
- 已经支持 `reranker_server_manager` 参数
- `_handle_processing_tools_state` 中已经有 reranker 调用逻辑

### 2. 为什么需要 SearchR1DualAgentLoopManager？

**管理两组 server**:
- 需要同时初始化 main agent 和 reranker 的 vLLM/SGLang servers
- 需要根据 `train_mode` 切换不同的生成逻辑
- 需要协调 wake_up/sleep 操作

### 3. 为什么复用 ExtendedAgentLoopWorker？

**避免重复代码**:
- Counterfactual rollout 逻辑复杂
- Worker 管理两组 server handles
- 已经实现了所有需要的功能

### 4. Hybrid Mode vs Standalone Mode

**Reranker Server 初始化**:
```python
# Hybrid mode (推荐)
server.init_hybrid(self.reranker_worker_group)
```

**为什么用 Hybrid？**
- 即使 reranker 不训练（inference-only），也可以用 hybrid mode
- Hybrid mode 只是表示 server 和 trainer 在同一个 Ray cluster
- 不影响是否训练，只影响资源共享方式

**Inference-only Reranker**:
- `trainable=false` → 使用 `Role.RerankerRollout`
- `reranker_steps=0` → 永不更新
- 但仍然可以用 hybrid mode 初始化 server

---

## agent_loop_config_path 说明

```yaml
# agent_loop_config.yaml
- name: search_r1_agent
  _target_: verl.experimental.agent_loop.search_r1_agent_loop.SearchR1AgentLoop
  # ... other configs ...
```

**作用**:
- 通过 Hydra 动态注册 agent loop 类
- 允许在 config 中指定使用哪个 agent loop 实现
- 支持自定义 agent loop 类

**在代码中的使用**:
```python
# agent_loop.py AgentLoopWorkerBase.__init__()
agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
if agent_loop_config_path:
    resolved_path = resolve_config_path(agent_loop_config_path)
    agent_loop_configs = OmegaConf.load(resolved_path)
    for agent_loop_config in agent_loop_configs:
        _agent_loop_registry[agent_loop_config.name] = agent_loop_config
```

**然后在 generate_sequences 中**:
```python
agent_name = trajectory.get("agent_name", "search_r1_agent")
agent_loop = hydra.utils.instantiate(
    _agent_loop_registry[agent_name],
    trainer_config=_DummyConfig(self.config),
    server_manager=self.server_manager,
    tokenizer=self.tokenizer,
    processor=self.processor,
)
```

---

## 总结

**核心架构**:
```
DualAgentRayTrainer
    ↓
SearchR1DualAgentLoopManager (管理两组 server)
    ↓
ExtendedAgentLoopWorker (处理 generation)
    ↓
SearchR1AgentLoop (实际的 agent 逻辑)
```

**训练流程**:
1. Trainer 调用 `manager.generate_sequences(batch, train_mode=...)`
2. Manager 根据 train_mode 选择：
   - "search_r1" → 标准生成
   - "reranker" → Counterfactual rollout
3. Worker 实例化 `SearchR1AgentLoop` 并传入相应的 server managers
4. AgentLoop 执行实际的 generation 和 tool calling

**优势**:
- ✅ 最大化复用现有代码
- ✅ 清晰的职责分离
- ✅ 灵活的训练模式切换
- ✅ 支持 inference-only reranker
