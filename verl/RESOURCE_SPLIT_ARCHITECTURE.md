# Dual Agent Resource Split Architecture

## Overview

Simple and reliable dual agent setup: **split GPUs into two independent groups**, each running a complete agent with its own vLLM servers and FSDP training.

## Architecture

### GPU Allocation (16 GPUs total)

```
Main Agent Group: GPUs 0-7
├── 8 GPUs for FSDP training (8-way sharding)
├── vLLM inference: 4 replicas × tensor_parallel_size=2
└── Independent Ray worker group

Reranker Group: GPUs 8-15  
├── 8 GPUs for FSDP training (8-way sharding)
├── vLLM inference: 4 replicas × tensor_parallel_size=2
└── Independent Ray worker group
```

### Resource Pool Configuration

```python
resource_pool_spec = {
    "main_agent_pool": [0, 1, 2, 3, 4, 5, 6, 7],      # GPUs 0-7
    "reranker_pool": [8, 9, 10, 11, 12, 13, 14, 15],  # GPUs 8-15
}

mapping = {
    Role.ActorRollout: "main_agent_pool",
    Role.RefPolicy: "main_agent_pool",
    Role.RerankerActorRollout: "reranker_pool",
    Role.RerankerRefPolicy: "reranker_pool",
}
```

### Training Strategy

**Time-division multiplexing**: Train agents sequentially, not concurrently.

```python
# Training loop
for batch in dataloader:
    if train_mode == "search_r1":
        # 1. Generate with main agent (uses GPUs 0-7)
        rollout_main = main_agent.rollout(batch)
        
        # 2. Reranker inference (uses GPUs 8-15)
        rollout_reranked = reranker.inference(rollout_main)
        
        # 3. Train main agent on GPUs 0-7
        main_agent.train(rollout_reranked)
        
    elif train_mode == "reranker":
        # 1. Generate with main agent (fixed, uses GPUs 0-7)
        rollout_main = main_agent.rollout(batch)
        
        # 2. Reranker counterfactual rollout (uses GPUs 8-15)
        rollout_counterfactual = reranker.counterfactual_rollout(rollout_main)
        
        # 3. Train reranker on GPUs 8-15  
        reranker.train(rollout_counterfactual)
```

## Benefits

✅ **No memory contention**: Each agent has dedicated GPUs  
✅ **Simple initialization**: Two independent worker groups spawn separately  
✅ **Predictable performance**: No context switching overhead  
✅ **Easy debugging**: Each agent is isolated  
✅ **Resource efficiency**: Both agents can work in parallel (rollout + inference)  

## Trade-offs

⚠️ **Half resources per agent**: Each agent uses only 8 GPUs instead of 16  
⚠️ **Cannot share model weights**: Need separate copies in GPU memory  
✅ **But**: More reliable and easier to maintain than FusedWorker approach  

## Implementation Changes

1. **ResourcePoolManager**: Define two separate resource pools with GPU split
2. **Worker initialization**: Create independent worker groups for each pool
3. **AgentLoopManager**: Remove GPU partitioning logic (not needed)
4. **Training loop**: Ensure agents train sequentially

## Expected Performance

- **Initialization**: ~6-10 minutes (4 replicas per agent, sequential)
- **Throughput**: Similar to FusedWorker (no context switching needed)
- **Memory**: ~8-10GB per GPU (fits within 80GB A100)
- **Stability**: High (no concurrent resource access)
