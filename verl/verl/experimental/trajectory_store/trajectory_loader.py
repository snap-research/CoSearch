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
TrajectoryLoader — loads Phase 1 trajectories from disk and extracts reranker training samples.

Each training sample for Phase 2 consists of:
  - initial_query: the user's original question
  - sub_query: the search query at this tool call step
  - top_50_documents: the retrieved documents (input for reranker)
  - trajectory_prefix: all messages up to (but not including) this tool response
  - answers: ground truth answers (for reward computation)
  - trajectory_uid: unique ID of the source trajectory
  - tool_call_step_index: which tool call in the trajectory this sample comes from
"""

import gzip
import json
import logging
import os
from typing import Any, Iterator

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrajectoryLoader:
    """Loads trajectory files and extracts reranker training samples.

    A single trajectory with K tool calls produces K training samples.
    Each sample provides:
      - The data needed to build a reranker prompt (initial_query, sub_query, top-50 docs)
      - The trajectory prefix for Search-R1 continuation after reranking
      - Ground truth answers for reward computation

    Usage:
        loader = TrajectoryLoader(trajectory_dir="/path/to/trajectories")
        samples = loader.load_all_samples()
        # or iterate lazily:
        for sample in loader.iter_samples():
            ...
    """

    def __init__(self, trajectory_dir: str, step_range: tuple[int, int] = None):
        """Initialize the trajectory loader.

        Args:
            trajectory_dir: Directory containing step_*.jsonl[.gz] files.
            step_range: Optional (start_step, end_step) to load a subset of steps.
                        Both endpoints are inclusive. If None, loads all.
        """
        self.trajectory_dir = trajectory_dir
        self.step_range = step_range
        self._files = self._discover_files()
        logger.info(
            f"[TrajectoryLoader] Found {len(self._files)} trajectory files in {trajectory_dir}"
        )

    def _discover_files(self) -> list[tuple[int, str]]:
        """Discover trajectory files and sort by step number.

        Returns:
            List of (step_number, filepath) tuples, sorted by step.
        """
        files = []
        if not os.path.isdir(self.trajectory_dir):
            logger.warning(f"[TrajectoryLoader] Directory not found: {self.trajectory_dir}")
            return files

        for fname in os.listdir(self.trajectory_dir):
            if not fname.startswith("step_"):
                continue
            # Parse step number: step_42.jsonl.gz → 42
            try:
                step_str = fname.split("_")[1].split(".")[0]
                step = int(step_str)
            except (IndexError, ValueError):
                logger.warning(f"[TrajectoryLoader] Skipping file with unparseable name: {fname}")
                continue

            if self.step_range is not None:
                start, end = self.step_range
                if step < start or step > end:
                    continue

            files.append((step, os.path.join(self.trajectory_dir, fname)))

        files.sort(key=lambda x: x[0])
        return files

    def iter_trajectories(self) -> Iterator[dict[str, Any]]:
        """Iterate over all raw trajectories from all step files.

        Yields:
            Raw trajectory dicts as saved by TrajectorySaver.
        """
        for step, filepath in self._files:
            open_fn = gzip.open if filepath.endswith(".gz") else open
            mode = "rt" if filepath.endswith(".gz") else "r"
            with open_fn(filepath, mode, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"[TrajectoryLoader] JSON decode error in {filepath}:{line_num}: {e}"
                        )
                        continue

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        """Iterate over reranker training samples extracted from trajectories.

        Each trajectory with K tool calls produces K samples.

        Yields:
            Sample dict with keys:
                - initial_query: str
                - sub_query: str
                - top_50_documents: list[dict]
                - top_5_documents: list[dict] (original top-5 for reference)
                - trajectory_prefix: list[dict] (messages up to this tool call)
                - answers: list[str]
                - final_reward: float (trajectory-level reward)
                - tool_score: float (this tool call's score)
                - answer_in_docs: bool
                - trajectory_uid: str
                - tool_call_step_index: int
                - trajectory_step: int (global training step)
        """
        for traj in self.iter_trajectories():
            samples = self._extract_samples_from_trajectory(traj)
            yield from samples

    def load_all_samples(self) -> list[dict[str, Any]]:
        """Load all reranker training samples into memory.

        Returns:
            List of sample dicts (see iter_samples for format).
        """
        return list(self.iter_samples())

    @staticmethod
    def _extract_samples_from_trajectory(traj: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract reranker training samples from a single trajectory.

        For each tool call in the trajectory, we produce one training sample containing:
        - The reranker input data (initial_query, sub_query, top-50 docs)
        - The trajectory prefix (messages up to this tool response)
        - Ground truth answers for reward computation

        The trajectory_prefix is the list of messages from the start of the conversation
        up to AND INCLUDING the assistant message that triggered this tool call.
        The tool response itself is NOT included — it will be replaced by the reranker's
        new output during Phase 2 training.

        Args:
            traj: Raw trajectory dict from TrajectorySaver.

        Returns:
            List of sample dicts.
        """
        samples = []
        messages = traj.get("messages", [])
        tool_calls = traj.get("tool_calls", [])

        if not tool_calls:
            return samples

        # Build trajectory prefixes by finding tool response positions in messages.
        # Message structure: [system?, user, assistant, tool, assistant, tool, ...]
        # We need to find each "tool" message and take all messages before it as prefix.
        tool_message_indices = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "tool":
                tool_message_indices.append(i)

        for tc_idx, tool_call in enumerate(tool_calls):
            # Skip tool calls without top-50 documents (not useful for reranker training)
            if not tool_call.get("top_50_documents"):
                continue

            # The trajectory prefix is all messages up to (but NOT including) the tool response.
            # This means: [system?, user, assistant_1, tool_1, ..., assistant_k]
            # where assistant_k is the one that triggered this tool call.
            if tc_idx < len(tool_message_indices):
                # Take everything before the tool response message
                prefix_end = tool_message_indices[tc_idx]
                trajectory_prefix = messages[:prefix_end]
            else:
                # Fallback: if we can't find the tool message, use all messages
                logger.warning(
                    f"[TrajectoryLoader] Could not find tool message index for tool call {tc_idx} "
                    f"in trajectory {traj.get('uid', 'unknown')}"
                )
                trajectory_prefix = messages

            samples.append({
                "initial_query": traj.get("initial_query", ""),
                "sub_query": tool_call.get("sub_query", ""),
                "top_50_documents": tool_call.get("top_50_documents", []),
                "top_5_documents": tool_call.get("top_5_documents", []),
                "trajectory_prefix": trajectory_prefix,
                "answers": traj.get("answers", []),
                "final_reward": traj.get("final_reward", 0.0),
                "tool_score": tool_call.get("tool_score", 0.0),
                "answer_in_docs": tool_call.get("answer_in_docs", False),
                "trajectory_uid": traj.get("uid", ""),
                "tool_call_step_index": tool_call.get("step_index", tc_idx),
                "trajectory_step": traj.get("step", 0),
            })

        return samples
