# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
TrajectorySaver — saves Phase 1 training trajectories to disk for Phase 2 reranker training.

Each training step produces a JSONL file (optionally gzip-compressed) containing one rollout per line.
Each rollout includes:
  - initial_query, answers, complete messages
  - per-tool-call details: sub_query, top_50_documents, top_5_documents, tool_score, answer_in_docs
  - final_reward, num_turns
"""

import gzip
import json
import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrajectorySaver:
    """Saves Phase 1 trajectories (with top-50 documents) to disk.

    File format: one JSONL file per training step.
      - step_0.jsonl.gz  (or step_0.jsonl if compress=False)
      - step_1.jsonl.gz
      - ...

    Each line is a JSON object representing one rollout trajectory.

    Usage:
        saver = TrajectorySaver(output_dir="/path/to/trajectories", compress=True)
        saver.save_step(step=0, trajectories=[...])
    """

    def __init__(self, output_dir: str, compress: bool = True):
        """Initialize the trajectory saver.

        Args:
            output_dir: Directory to save trajectory files.
            compress: Whether to gzip-compress the output files.
        """
        self.output_dir = output_dir
        self.compress = compress
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"[TrajectorySaver] Saving trajectories to {output_dir} (compress={compress})")

    def save_step(self, step: int, trajectories: list[dict[str, Any]]) -> str:
        """Save all trajectories from a single training step.

        Args:
            step: Global training step number.
            trajectories: List of trajectory dicts, each containing:
                - step: int
                - uid: str
                - initial_query: str
                - answers: list[str]
                - messages: list[dict]
                - tool_calls: list[dict] with sub_query, top_50_documents, etc.
                - final_reward: float
                - num_turns: int

        Returns:
            Path to the saved file.
        """
        suffix = ".jsonl.gz" if self.compress else ".jsonl"
        filename = f"step_{step}{suffix}"
        filepath = os.path.join(self.output_dir, filename)

        open_fn = gzip.open if self.compress else open
        mode = "wt" if self.compress else "w"

        with open_fn(filepath, mode, encoding="utf-8") as f:
            for traj in trajectories:
                line = json.dumps(traj, ensure_ascii=False, default=self._json_serializer)
                f.write(line + "\n")

        logger.info(
            f"[TrajectorySaver] Saved {len(trajectories)} trajectories for step {step} → {filepath}"
        )
        return filepath

    @staticmethod
    def build_trajectory_from_rollout(
        step: int,
        uid: str,
        initial_query: str,
        answers: list[str],
        messages: list[dict[str, Any]],
        tool_call_details: list[dict[str, Any]],
        final_reward: float,
        num_turns: int,
    ) -> dict[str, Any]:
        """Build a trajectory dict from rollout outputs.

        This is a helper to construct the standard trajectory format
        from the data available in the agent loop after a rollout completes.

        Args:
            step: Global training step.
            uid: Unique rollout ID.
            initial_query: User's original question.
            answers: Ground truth answer strings.
            messages: Complete conversation messages list.
            tool_call_details: List of per-tool-call info dicts, each containing:
                - step_index: int (which tool call in the trajectory)
                - sub_query: str
                - top_50_documents: list[dict] (from retrieval)
                - top_5_documents: list[dict] (final returned docs)
                - tool_score: float
                - answer_in_docs: bool
            final_reward: Final trajectory reward score.
            num_turns: Number of conversation turns.

        Returns:
            Trajectory dict ready for save_step().
        """
        return {
            "step": step,
            "uid": uid,
            "initial_query": initial_query,
            "answers": answers,
            "messages": messages,
            "tool_calls": tool_call_details,
            "final_reward": final_reward,
            "num_turns": num_turns,
        }

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for numpy/torch types."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # torch tensors
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
