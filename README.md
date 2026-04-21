# CoSearch: Joint Training of Reasoning and Document Ranking via Reinforcement Learning for Agentic Search

CoSearch jointly trains a **multi-step reasoning agent** and a **generative document ranker** via GRPO for agentic search. The main agent issues sub-queries; the ranker reorders candidate documents from a fixed dense retriever before the agent observes them — both are optimized end-to-end from answer correctness.

Two technical contributions make this work:
- **Semantic grouping**: clusters sub-queries by token-level F1 similarity to form valid GRPO groups for the ranker, improving sampling efficiency without additional rollouts.
- **Composite reward**: combines a ranking quality signal (Hit@k) with trajectory-level answer correctness to give the ranker both immediate and long-term learning signals.

## Step 1: Set Up Environments
### Training environment (`search-llm`)

```bash
bash conda_setup/setup_conda_env.sh
conda activate search-llm
```

See [conda_setup/README.md](conda_setup/README.md) for optional flags (CUDA version, force-recreate, skip flash-attn).

### Retriever environment (`retriever`)

We use e5-base as the retriever. The retrieval server setup follows [Search-R1](https://github.com/PeterGriffinJin/Search-R1), which is already cloned at `Search-R1/`. You can create the conda environment with:

```bash
conda create -n retriever python=3.10
conda activate retriever

# Install torch with conda (needed for faiss-gpu)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets

# faiss-gpu for efficient retrieval
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# FastAPI server
pip install uvicorn fastapi
```

---

## Step 2: Download Retrieval Index and Corpus

The retriever uses a Wikipedia passage index (e5-base-v2 embeddings). Download from Search-R1:

```bash
save_path=/your/data/path

cd Search-R1
python scripts/download.py --save_path $save_path

# Merge split index files
cat $save_path/part_* > $save_path/e5_Flat.index

# Decompress corpus
gzip -d $save_path/wiki-18.jsonl.gz
```

---

## Step 3: Launch the Retrieval Server

The retrieval server must be running before training starts. It exposes a `/retrieve` endpoint on port 8000.

**Edit** `scripts/launch_retriever_server.sh` and set your data paths:

```bash
INDEX_FILE="/path/to/e5_Flat.index"
CORPUS_FILE="/path/to/wiki-18.jsonl"
RETRIEVER_MODEL="intfloat/e5-base-v2"
TOPK=50
```

Then submit:

```bash
sbatch scripts/launch_retriever_server.sh
```

The server will log its hostname to `sbatch_out/retriever_<jobid>.out`. Note the hostname (e.g., `gpu013`) — you will need it for training.

> **Tip:** You can also launch the server interactively (without sbatch):
> ```bash
> conda activate retriever
> bash scripts/launch_retriever_server.sh
> ```

---

## Step 4: Download CoSearch Training and EvaluaDon ata

Before launching training, download the train/eval parquet files to the local `data/` path expected by the training pipeline:

```bash
python scripts/download_from_hf.py
```

By default, this pulls from:
- `hzeng/co-search-train`
- `hzeng/co-search-eval`

If needed, override the dataset repos:

```bash
python scripts/download_from_hf.py --train-repo <train_repo> --eval-repo <eval_repo>
```

---

## Step 5: Launch CoSearch Training

Pass the retriever URL as an environment variable at submission time:

```bash
RETRIEVAL_SERVICE_URL="http://<retriever-hostname>:8000/retrieve" sbatch scripts/train_co_search_grpo.sh
```

The retriever hostname comes from the `sbatch_out/retriever_<jobid>.out` log from Step 3.

The script generates the tool config at runtime with this URL injected — no manual file editing needed. The default is `http://localhost:8000/retrieve` if `RETRIEVAL_SERVICE_URL` is not set.

---

## Citation

If you use CoSearch in your research, please cite:

- https://arxiv.org/abs/2604.17555

```bibtex
@article{zeng2026cosearch,
	title={CoSearch: Joint Training of Reasoning and Document Ranking via Reinforcement Learning for Agentic Search},
	author={Hansi Zeng, Liam Collins, Bhuvesh Kumar, Neil Shah, Hamed Zamani},
	journal={arXiv preprint arXiv:2604.17555},
	year={2026},
	doi={10.48550/arXiv.2604.17555}
}
```

---

## Authorship

This repository contains sample code developed as part of a collaboration between Snap Inc. and the University of Massachusetts Amherst. Rights to the sample code remain with the original author(s) and are licensed under the terms described in the LICENSE file.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). See the [LICENSE](LICENSE) file for details.
