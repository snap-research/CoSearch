#!/usr/bin/env bash
# Build a conda env that mirrors Dockerfile.stable.vllm as closely as possible.

set -euo pipefail

ENV_NAME="${ENV_NAME:-search-llm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTORCH_CUDA_TAG="${PYTORCH_CUDA_TAG:-cu128}" # Example: cu124 / cu128
FORCE_RECREATE="${FORCE_RECREATE:-0}"          # 1 => recreate env
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}" # 1(required) / auto / 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found. Install Miniconda/Anaconda first."
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    if [ "${FORCE_RECREATE}" = "1" ]; then
        echo "=== Removing existing env: ${ENV_NAME} ==="
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "=== Reusing existing env: ${ENV_NAME} ==="
    fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "=== Creating conda env: ${ENV_NAME} (python ${PYTHON_VERSION}) ==="
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel

echo "=== Install PyTorch (${PYTORCH_CUDA_TAG}) ==="
python -m pip install --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA_TAG}" \
    torch torchvision torchaudio

echo "=== Install vLLM + pinned tokenizer stack ==="
python -m pip install --no-cache-dir \
    "vllm==0.11.0" \
    "transformers==4.57.6" \
    "tokenizers==0.22.2"

ensure_cuda_env_for_flash_attn() {
    # Try to initialize environment-modules when running in non-interactive shells.
    if ! command -v module >/dev/null 2>&1; then
        if [ -f /etc/profile.d/modules.sh ]; then
            # shellcheck disable=SC1091
            source /etc/profile.d/modules.sh
        fi
    fi

    if command -v nvcc >/dev/null 2>&1; then
        return 0
    fi

    # Common cluster module names/versions; stop at first success.
    if command -v module >/dev/null 2>&1; then
        for cuda_mod in cuda/12.8 cuda/12.6 cuda/12.4.1 cuda/12.4 cuda/12.1 cuda/11.8 cuda; do
            if module load "${cuda_mod}" >/dev/null 2>&1; then
                if command -v nvcc >/dev/null 2>&1; then
                    echo "Loaded CUDA module: ${cuda_mod}"
                    return 0
                fi
            fi
        done
    fi

    return 1
}

echo "=== Install flash-attn (mode: ${INSTALL_FLASH_ATTN}) ==="
detect_cuda_home() {
    if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
        echo "${CUDA_HOME}"
        return 0
    fi

    if command -v nvcc >/dev/null 2>&1; then
        local nvcc_path nvcc_bin cuda_home_guess
        nvcc_path="$(command -v nvcc)"
        nvcc_bin="$(dirname "${nvcc_path}")"
        cuda_home_guess="$(cd "${nvcc_bin}/.." && pwd)"
        if [ -x "${cuda_home_guess}/bin/nvcc" ]; then
            echo "${cuda_home_guess}"
            return 0
        fi
    fi

    if [ -x "/usr/local/cuda/bin/nvcc" ]; then
        echo "/usr/local/cuda"
        return 0
    fi

    return 1
}

install_flash_attn() {
    python -m pip install --no-cache-dir --no-build-isolation "flash_attn==2.8.1"
}

if [ "${INSTALL_FLASH_ATTN}" = "0" ]; then
    echo "Skip flash-attn installation (INSTALL_FLASH_ATTN=0)."
elif [ "${INSTALL_FLASH_ATTN}" = "1" ]; then
    ensure_cuda_env_for_flash_attn || true
    if CUDA_HOME_DETECTED="$(detect_cuda_home)"; then
        export CUDA_HOME="${CUDA_HOME_DETECTED}"
        echo "Using CUDA_HOME=${CUDA_HOME}"
        install_flash_attn
    else
        echo "ERROR: INSTALL_FLASH_ATTN=1 but nvcc/CUDA_HOME not found."
        echo "Please load CUDA toolkit first (e.g. module load cuda/12.8) or set CUDA_HOME."
        exit 1
    fi
else
    # auto mode: install only when CUDA toolchain is available; otherwise continue.
    ensure_cuda_env_for_flash_attn || true
    if CUDA_HOME_DETECTED="$(detect_cuda_home)"; then
        export CUDA_HOME="${CUDA_HOME_DETECTED}"
        echo "Using CUDA_HOME=${CUDA_HOME}"
        install_flash_attn
    else
        echo "WARNING: nvcc/CUDA_HOME not found. Skipping flash-attn in auto mode."
        echo "If needed later: module load cuda && INSTALL_FLASH_ATTN=1 bash conda_setup/setup_conda_env.sh"
    fi
fi

echo "=== Install training/runtime deps ==="
python -m pip install --no-cache-dir \
    pybind11 \
    codetiming \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    mathruler \
    pylatexenc \
    qwen_vl_utils \
    "trl==0.9.6" \
    nvtx \
    matplotlib \
    liger_kernel \
    datasets \
    uvicorn \
    fastapi \
    pydantic \
    pytrec-eval \
    ujson \
    openai \
    anyio \
    omegaconf \
    hydra-core \
    peft \
    accelerate \
    tensorboard \
    torchdata \
    wandb \
    "pyarrow>=19.0.0" \
    dill \
    pandas \
    "numpy<2.0.0" \
    "packaging>=20.0"

echo "=== Install Megatron-LM and mbridge ==="
python -m pip install --no-deps --no-cache-dir \
    "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.14.0rc7"
python -m pip install -U "git+https://github.com/ISEEKYAN/mbridge.git"

echo "=== Install local verl (editable) ==="
if [ -d "${PROJECT_ROOT}/verl" ]; then
    python -m pip install --no-deps -e "${PROJECT_ROOT}/verl"
else
    echo "WARNING: ${PROJECT_ROOT}/verl not found."
    echo "Install manually later:"
    echo "  pip install --no-deps -e verl"
fi

echo
echo "============================================"
echo "Conda environment ready: ${ENV_NAME}"
echo "Activate:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Quick sanity check:"
echo "  python -c \"import torch, vllm, verl; print(torch.__version__, vllm.__version__, verl.__version__)\""
echo "============================================"
echo
echo "Optional (advanced, source build): apex / TransformerEngine / DeepEP"
echo "Refer to Dockerfile.stable.vllm for exact build flags and versions."
