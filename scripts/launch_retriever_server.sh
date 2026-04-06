#!/bin/bash
#SBATCH --job-name=retriever-server
#SBATCH --output=sbatch_out/retriever_%j.out
#SBATCH -p gpu
#SBATCH -G 6
#SBATCH -C 2080ti
#SBATCH --nodes=1
#SBATCH --qos=long
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=7-00:00:00

# =============================================================================
# Retriever Server Launch Script
# Starts a FastAPI/faiss dense retrieval server used by CoSearch during RL training.
#
# Usage:
#   sbatch scripts/launch_retriever_server.sh
#
# The server listens on port 8000 and exposes a /retrieve endpoint.
# Set RETRIEVAL_SERVICE_URL in train_co_search_grpo.sh to point to this host:port.
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURE THESE BEFORE RUNNING
#
# How to get the index and corpus:
#   See README.md -> Step 2: Download Retrieval Index and Corpus
#
#   Short version:
#     save_path=/your/data/path
#     cd Search-R1
#     python scripts/download.py --save_path $save_path
#     cat $save_path/part_* > $save_path/e5_Flat.index
#     gzip -d $save_path/wiki-18.jsonl.gz
#
# Then set INDEX_FILE and CORPUS_FILE to the resulting paths below.
# =============================================================================
INDEX_FILE="/path/to/e5_Flat.index"
CORPUS_FILE="/path/to/wiki-18.jsonl"
RETRIEVER_MODEL="intfloat/e5-base-v2"
TOPK=50   # CoSearch fetches top-50 then reranks down to top-m

# Path to your conda installation (update if conda is elsewhere)
CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"

# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "${INDEX_FILE}" ]; then
    echo "ERROR: index file not found: ${INDEX_FILE}"
    echo "See README.md -> Step 2 for download instructions."
    exit 1
fi

if [ ! -f "${CORPUS_FILE}" ]; then
    echo "ERROR: corpus file not found: ${CORPUS_FILE}"
    echo "See README.md -> Step 2 for download instructions."
    exit 1
fi

# Activate retriever conda environment
if [ ! -f "${CONDA_SH}" ]; then
    echo "ERROR: conda not found at ${CONDA_SH}"
    echo "Update the CONDA_SH variable in this script."
    exit 1
fi
. "${CONDA_SH}"
conda activate retriever

echo "=== Starting retrieval server ==="
echo "Index:   ${INDEX_FILE}"
echo "Corpus:  ${CORPUS_FILE}"
echo "Model:   ${RETRIEVER_MODEL}"
echo "Top-k:   ${TOPK}"
echo "Server will listen on 0.0.0.0:8000"

python "${PROJECT_ROOT}/Search-R1/search_r1/search/retrieval_server.py" \
    --index_path "${INDEX_FILE}" \
    --corpus_path "${CORPUS_FILE}" \
    --topk "${TOPK}" \
    --retriever_model "${RETRIEVER_MODEL}"
