# Conda Environment Setup

One-command setup for the `search-llm` training environment.

## Quick Start

From the CoSearch project root:

```bash
bash conda_setup/setup_conda_env.sh
conda activate search-llm
```

Optional flags:

```bash
# Force-recreate if the env already exists
FORCE_RECREATE=1 bash conda_setup/setup_conda_env.sh

# Different CUDA version (e.g. cu121 for CUDA 12.1)
PYTORCH_CUDA_TAG=cu121 bash conda_setup/setup_conda_env.sh

# Skip flash-attn (e.g. no nvcc available)
INSTALL_FLASH_ATTN=0 bash conda_setup/setup_conda_env.sh
```

## Core Versions

| Package | Version |
|---------|---------|
| vllm | 0.11.0 |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| flash_attn | 2.8.1 |
| trl | 0.9.6 |
| verl | local editable install from `verl/` |

## Sanity Check

```bash
python -c "import torch, vllm, verl; print(torch.__version__, vllm.__version__, verl.__version__)"
python -c "import flash_attn; print('flash_attn ok')"
```

## Common Issues

- **flash_attn build fails**: CUDA toolkit (`nvcc`) not on PATH. Load it first: `module load cuda/12.8`
- **slow HuggingFace downloads**: Set `HF_HOME` to a fast shared disk before running training
- **multi-GPU / Ray issues**: Ensure your Slurm job requests the correct GPU count and `CUDA_VISIBLE_DEVICES` is set properly
