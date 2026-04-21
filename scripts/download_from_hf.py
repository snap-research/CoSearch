"""
Download CoSearch dataset from HuggingFace Hub and save as parquet
to the original local paths (lossless roundtrip).

Usage:
    python scripts/download_from_hf.py
    python scripts/download_from_hf.py --train-repo hzeng/co-search-train --eval-repo hzeng/co-search-eval
"""

import argparse
from pathlib import Path

from datasets import load_dataset

TRAIN_PARQUET = Path("data/co_search/nq_40_multihop_60_51K/cot/co_search_rl_51k.train.parquet")
EVAL_PARQUET  = Path("data/co_search/nq_40_multihop_60_51K/cot/co_search_26k.sample_eval.parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-repo",
        default="hzeng/co-search-train",
        help="HuggingFace dataset repo id for train split, e.g. hzeng/co-search-train",
    )
    parser.add_argument(
        "--eval-repo",
        default="hzeng/co-search-eval",
        help="HuggingFace dataset repo id for eval split, e.g. hzeng/co-search-eval",
    )
    args = parser.parse_args()

    print(f"Downloading train split from {args.train_repo}...")
    train_ds = load_dataset(args.train_repo)

    print(f"Downloading eval split from {args.eval_repo}...")
    eval_ds = load_dataset(args.eval_repo)

    TRAIN_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving train split -> {TRAIN_PARQUET}")
    train_ds["train"].to_parquet(str(TRAIN_PARQUET))

    print(f"Saving eval split  -> {EVAL_PARQUET}")
    eval_ds["train"].to_parquet(str(EVAL_PARQUET))

    print(f"Done.")
    print(f"  train: {len(train_ds['train']):,} rows")
    print(f"  eval:  {len(eval_ds['train']):,} rows")


if __name__ == "__main__":
    main()
