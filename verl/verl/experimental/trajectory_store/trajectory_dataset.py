# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2024 The verl Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""
Reranker Phase-2 Training Dataset.

Loads Phase-1 trajectory samples, applies sub-query frequency filtering,
pre-builds ``raw_prompt`` (chat-message list) and ``docid_map`` so that
the downstream worker receives everything it needs from the batch.

Filtering algorithm (per the GRPO-repeat redesign):
  1. Group samples by ``trajectory_uid``.
  2. Within each group, count sub_query frequency.
  3. Keep the top-K most frequent sub_queries per trajectory_uid.
  4. For each (trajectory_uid, sub_query) pair, randomly pick **one** instance.
  5. From the de-duplicated set, randomly sample ``total_samples`` entries.
"""

from __future__ import annotations

import logging
import os
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from verl.experimental.trajectory_store.trajectory_loader import TrajectoryLoader
from verl.tools.utils.prompts import RERANK_PROMPT_WITH_INITIAL_QUERY
from verl.tools.utils.search import format_tool_response_with_docid_map

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class RerankerTrainingDataset(Dataset):
    """Dataset that reads Phase-1 trajectory outputs and produces per-sample
    dicts ready for GRPO training of the reranker.

    Each sample contains ``raw_prompt`` (list[dict] chat messages) and
    ``docid_map`` (dict mapping 1-based index → doc dict) in addition to
    the original trajectory fields.
    """

    def __init__(
        self,
        trajectory_dir: str,
        step_range: tuple[int, int] | None = None,
        min_documents: int = 5,
        top_k_sub_queries: int = 4,
        total_samples: int | None = None,
        reranker_top_m: int = 5,
        seed: int = 42,
    ):
        """
        Args:
            trajectory_dir: Path to the directory containing trajectory JSONL files.
            step_range: (start_step, end_step) inclusive. ``None`` means all steps.
            min_documents: Minimum number of retrieved documents required per sample.
            top_k_sub_queries: Keep at most this many most-frequent sub_queries
                per trajectory_uid.  Set to 0 or ``None`` to disable filtering.
            total_samples: Final dataset size after random sampling. ``None``
                means keep all surviving samples (no down-sampling).
            reranker_top_m: Number of top documents the reranker should output
                (``M`` in the prompt template).
            seed: Random seed for reproducible sampling.
        """
        super().__init__()
        self.trajectory_dir = trajectory_dir
        self.step_range = step_range
        self.min_documents = min_documents
        self.top_k_sub_queries = top_k_sub_queries
        self.total_samples = total_samples
        self.reranker_top_m = reranker_top_m
        self.seed = seed

        self.samples: list[dict[str, Any]] = []
        self._load_and_filter()

    # ------------------------------------------------------------------
    # Loading & filtering
    # ------------------------------------------------------------------

    def _load_and_filter(self) -> None:
        """Load trajectory samples, apply filters, build prompts."""
        loader = TrajectoryLoader(
            trajectory_dir=self.trajectory_dir,
            step_range=self.step_range,
        )

        # Stage 1: basic quality filter
        raw_samples: list[dict[str, Any]] = []
        skipped_negative_reward = 0
        skipped_few_docs = 0
        for sample in loader.iter_samples():
            # Skip trajectories with negative final_reward (format errors in Search-R1)
            if sample.get("final_reward", 0.0) < 0:
                skipped_negative_reward += 1
                continue
            docs = sample.get("top_50_documents", [])
            if len(docs) < self.min_documents:
                skipped_few_docs += 1
                continue
            raw_samples.append(sample)

        logger.info(
            f"[RerankerTrainingDataset] Loaded {len(raw_samples)} samples "
            f"(min_documents={self.min_documents}, "
            f"skipped_negative_reward={skipped_negative_reward}, "
            f"skipped_few_docs={skipped_few_docs})"
        )

        if not raw_samples:
            logger.warning("[RerankerTrainingDataset] No samples after quality filter!")
            return

        # Stage 2: sub-query frequency filtering
        filtered = self._filter_by_sub_query_frequency(raw_samples)

        logger.info(
            f"[RerankerTrainingDataset] After sub-query filtering: {len(filtered)} samples "
            f"(top_k={self.top_k_sub_queries})"
        )

        # Stage 3: random down-sampling to total_samples
        rng = random.Random(self.seed)
        if self.total_samples is not None and len(filtered) > self.total_samples:
            filtered = rng.sample(filtered, self.total_samples)
            logger.info(
                f"[RerankerTrainingDataset] Down-sampled to {len(filtered)} samples"
            )

        # Stage 4: build raw_prompt + docid_map for every surviving sample
        for sample in filtered:
            self._attach_prompt(sample)

        self.samples = filtered
        logger.info(
            f"[RerankerTrainingDataset] Final dataset size: {len(self.samples)}"
        )

    def _filter_by_sub_query_frequency(
        self, samples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply the top-K sub-query frequency filter.

        1. Group by ``trajectory_uid``.
        2. Count sub_query occurrences per group.
        3. Keep top-K most frequent sub_queries per uid.
        4. For each (uid, sub_query) pair keep ONE random instance.
        """
        if not self.top_k_sub_queries or self.top_k_sub_queries <= 0:
            return samples

        rng = random.Random(self.seed)

        # Group by trajectory_uid
        uid_groups: dict[str, list[dict]] = defaultdict(list)
        for s in samples:
            uid_groups[s["trajectory_uid"]].append(s)

        result: list[dict[str, Any]] = []
        for uid, group in uid_groups.items():
            # Count sub_query frequency within this trajectory
            sq_counter = Counter(s["sub_query"] for s in group)
            top_sqs = {
                sq for sq, _count in sq_counter.most_common(self.top_k_sub_queries)
            }

            # Group by sub_query, keep only top-K sub_queries
            sq_to_samples: dict[str, list[dict]] = defaultdict(list)
            for s in group:
                if s["sub_query"] in top_sqs:
                    sq_to_samples[s["sub_query"]].append(s)

            # Pick one random instance per (uid, sub_query)
            for sq, sq_samples in sq_to_samples.items():
                result.append(rng.choice(sq_samples))

        return result

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _attach_prompt(self, sample: dict[str, Any]) -> None:
        """Build ``raw_prompt`` (chat-message list) and ``docid_map`` in-place.

        Uses the same logic as ``_build_reranker_prompt`` in the worker so that
        the prompt is deterministic and independent of rollout index.
        """
        documents = sample["top_50_documents"]
        initial_query = sample.get("initial_query", "")
        sub_query = sample["sub_query"]
        top_m = self.reranker_top_m

        passage_block, docid_map = format_tool_response_with_docid_map(documents)
        prompt_text = RERANK_PROMPT_WITH_INITIAL_QUERY.format(
            N=len(documents),
            M=min(top_m, len(documents)),
            initial_query=initial_query,
            sub_query=sub_query,
            passages_block=passage_block,
        )

        # raw_prompt is the standard verl chat-message list
        sample["raw_prompt"] = [{"role": "user", "content": prompt_text}]
        sample["docid_map"] = docid_map
        sample["top_m"] = top_m

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, list]:
        """Collate a list of sample dicts into a dict of lists.

        Keys are union of all sample keys; missing values become None.
        """
        all_keys = set()
        for sample in batch:
            all_keys.update(sample.keys())

        collated: dict[str, list] = {key: [] for key in all_keys}
        for sample in batch:
            for key in all_keys:
                collated[key].append(sample.get(key))

        return collated
