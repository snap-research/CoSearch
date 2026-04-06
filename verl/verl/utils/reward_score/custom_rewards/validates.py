"""
Validation Functions for Reranker Output

This module contains validation logic for reranker LLM outputs.

SOURCE: This code is copied from:
    /work/hzeng_umass_edu/ir-research/search-embeded-llm/search_llm/retriever/reranker/async_rerank_engine.py
    
The validation functions (validate_qwen2507_instruct_thinking_output, etc.) are 
extracted from the original async_rerank_engine.py to avoid complex import dependencies
in the VERL training environment.

If you need to update validation logic, please update both files:
    1. search_llm/retriever/reranker/async_rerank_engine.py (source of truth)
    2. verl/verl/utils/reward_score/custom_rewards/validates.py (this file - copy for VERL)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import re


# Regular expressions for parsing output blocks
# SOURCE: From async_rerank_engine.py lines 416-420
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.S)
REASON_BLOCK_RE = re.compile(r"<reason>(.*?)</reason>", re.S)
SEL_BLOCK_RE = re.compile(r"<select>(.*?)</select>", re.S)
RERANK_BLOCK_RE = re.compile(r"<rerank>(.*?)</rerank>", re.S)
IDX_RE = re.compile(r"\[(\d+)\]")


# SOURCE: From async_rerank_engine.py lines 423-425
def _parse_indices(text: str) -> List[int]:
    """Extract all indices in [n] format from text"""
    return [int(x) for x in IDX_RE.findall(text)]


# SOURCE: From async_rerank_engine.py lines 612-742
# Function: validate_qwen2507_instruct_thinking_output
def validate_qwen2507_instruct_thinking_output(
    output: str, N: int, M: int, id_to_pid: dict
):
    """
    Validate output from Qwen 2507 INSTRUCT models WITH thinking enabled
    
    Format expected:
        <reason>reasoning content</reason>
        <select>[i], [j], ...</select>
        <rerank>[i] > [j] > ...</rerank>
    
    Rules:
      - Must contain exactly ONE <reason> block
      - Validate ONLY the tail after </reason>
      - Tail must contain exactly ONE <select> and ONE <rerank> block
      - <select>: M indices in ascending order, comma-separated
      - <rerank>: same M indices in relevance order, '>' separated
    
    Args:
        output: Raw LLM output string
        N: Total number of candidate passages
        M: Number of passages to select and rerank
        id_to_pid: Mapping from indices to passage IDs
    
    Returns:
        Validation result dictionary
    """
    errors: List[str] = []
    error_type: str | None = None
    selected: List[int] = []
    reranked: List[int] = []

    text = output or ""

    # 1) <reason> 唯一性
    reason_blocks = list(REASON_BLOCK_RE.finditer(text))
    if len(reason_blocks) == 0:
        errors.append("Missing <reason> block.")
        error_type = error_type or "MissingReason"
    elif len(reason_blocks) > 1:
        errors.append(f"Expected exactly 1 <reason> block, got {len(reason_blocks)}.")
        error_type = error_type or "DuplicateReason"
    # 从最后一个 </reason> 之后取尾部（若没有 </reason> 则取整段，便于给出更多诊断）
    last_close = text.rfind("</reason>")
    tail = text[last_close + len("</reason>"):] if last_close != -1 else text
    tail = tail.strip()

    # 2) 尾部必须严格是 <select>... </select> + <rerank>... </rerank>
    strict_tail_pat = re.compile(
        r"^\s*<select>(?P<sel>.*?)</select>\s*<rerank>(?P<rank>.*?)</rerank>\s*$",
        re.S
    )
    m = strict_tail_pat.match(tail)
    if not m:
        errors.append("Tail after </reason> must contain only one <select>...</select> followed by one <rerank>...</rerank> (no extra text).")
        error_type = error_type or "BadTailOrder"

        # 进一步诊断数量
        tail_sel = SEL_BLOCK_RE.findall(tail)
        tail_rank = RERANK_BLOCK_RE.findall(tail)
        if len(tail_sel) == 0:
            errors.append("Missing <select> block in tail.")
            error_type = error_type or "MissingSelect"
        elif len(tail_sel) > 1:
            errors.append(f"Found {len(tail_sel)} <select> blocks in tail; expected exactly one.")
            error_type = error_type or "DuplicateSelect"

        if len(tail_rank) == 0:
            errors.append("Missing <rerank> block in tail.")
            error_type = error_type or "MissingRerank"
        elif len(tail_rank) > 1:
            errors.append(f"Found {len(tail_rank)} <rerank> blocks in tail; expected exactly one.")
            error_type = error_type or "DuplicateRerank"

        return {
            "status_message": f"Error: {error_type or 'ValidationError'}",
            "error_type": error_type or "ValidationError",
            "errors": errors,
            "selected": [],
            "reranked": [],
            "raw": output,
        }

    sel_text  = (m.group("sel") or "").strip()
    rank_text = (m.group("rank") or "").strip()

    # 3) 校验 <select>
    comma_list_re = re.compile(r"^\s*\[\d+\](\s*,\s*\[\d+\]){" + str(M - 1) + r"}\s*$")
    if not comma_list_re.match(sel_text):
        errors.append("Select list must be exactly M indices separated by commas (e.g., [3], [27], [105], ...).")
        error_type = error_type or "BadSelectCommaFormat"

    selected = _parse_indices(sel_text)

    if len(selected) != M:
        errors.append(f"<select> must contain exactly {M} indices, got {len(selected)}.")
        error_type = error_type or "BadSelectCount"

    if selected and selected != sorted(selected):
        errors.append("<select> indices must be in ascending order.")
        error_type = error_type or "SelectNotAscending"

    if selected and not all(1 <= i <= N for i in selected):
        errors.append("Some <select> indices are out of range [1..N].")
        error_type = error_type or "SelectOutOfRange"

    if selected and len(set(selected)) != len(selected):
        errors.append("<select> indices must be unique.")
        error_type = error_type or "SelectDuplicate"

    # 4) 校验 <rerank>
    chain_re = re.compile(r"^\s*\[\d+\](\s*>\s*\[\d+\]){" + str(M - 1) + r"}\s*$")
    if not chain_re.match(rank_text):
        gt_count = rank_text.count(">")
        if gt_count != (M - 1):
            errors.append(f"<rerank> must contain exactly {M-1} '>' separators, got {gt_count}.")
            error_type = error_type or "BadRerankGtCount"
        else:
            errors.append("Rerank chain must be [i] > [j] > ... with spaces only around '>'.")
            error_type = error_type or "BadRerankFormat"

    rerank_ids = _parse_indices(rank_text)

    if len(rerank_ids) != M:
        errors.append(f"<rerank> must list exactly {M} indices, got {len(rerank_ids)}.")
        error_type = error_type or "BadRerankCount"

    if selected and set(rerank_ids) != set(selected):
        errors.append("<rerank> must be a permutation of the <select> indices.")
        error_type = error_type or "RerankNotPermutation"

    # 5) 错误早返
    if errors:
        return {
            "status_message": f"Error: {error_type or 'ValidationError'}",
            "error_type": error_type or "ValidationError",
            "errors": errors,
            "selected": [],
            "reranked": [],
            "raw": output,
        }

    # 6) id_to_pid 映射
    missing = [i for i in rerank_ids if i not in id_to_pid]
    if missing:
        return {
            "status_message": "Error: MissingIdMap",
            "error_type": "MissingIdMap",
            "errors": [f"id_to_pid missing keys for indices: {missing}"],
            "selected": [],
            "reranked": [],
            "raw": output,
        }

    return {
        "status_message": "Success.",
        "error_type": None,
        "errors": [],
        "selected": selected,
        "reranked": [id_to_pid[i] for i in rerank_ids],
        "raw": output,
    }

def validate_qwen2507_direct_rerank_output(
    output: str, N: int, M: int, id_to_pid: dict
) -> Dict[str, Any]:
    """
    Validate output from Qwen 2507 INSTRUCT models with FULL_PSSG format (implicit N→M selection)
    
    Format expected:
        <reason>reasoning content</reason>
        <rerank>[i] > [j] > ...</rerank>
    
    Rules:
      - Must contain exactly ONE <reason> block
      - Must contain exactly ONE <rerank> block after </reason>
      - NO <select> block (selection is implicit in the rerank output)
      - <rerank>: exactly M indices in relevance order, '>' separated
      - All indices must be valid [1..N], unique, and mapped to passage IDs
    
    Args:
        output: Raw LLM output string
        N: Total number of candidate passages
        M: Number of passages to select and rerank (implicitly)
        id_to_pid: Mapping from indices to passage IDs
    
    Returns:
        Validation result dictionary with selected=reranked (since selection is implicit)
    """
    errors: List[str] = []
    error_type: str | None = None
    reranked: List[int] = []

    text = output or ""

    # 1) <reason> uniqueness check
    reason_blocks = list(REASON_BLOCK_RE.finditer(text))
    if len(reason_blocks) == 0:
        errors.append("Missing <reason> block.")
        error_type = error_type or "MissingReason"
    elif len(reason_blocks) > 1:
        errors.append(f"Expected exactly 1 <reason> block, got {len(reason_blocks)}.")
        error_type = error_type or "DuplicateReason"
    
    # Extract tail after last </reason>
    last_close = text.rfind("</reason>")
    tail = text[last_close + len("</reason>"):] if last_close != -1 else text
    tail = tail.strip()

    # 2) Tail must be exactly one <rerank>...</rerank> block (NO <select>)
    strict_tail_pat = re.compile(
        r"^\s*<rerank>(?P<rank>.*?)</rerank>\s*$",
        re.S
    )
    m = strict_tail_pat.match(tail)
    if not m:
        errors.append("Tail after </reason> must contain only one <rerank>...</rerank> block (no <select>, no extra text).")
        error_type = error_type or "BadTailOrder"

        # Diagnose what's in the tail
        tail_sel = SEL_BLOCK_RE.findall(tail)
        tail_rank = RERANK_BLOCK_RE.findall(tail)
        
        if len(tail_sel) > 0:
            errors.append(f"Found {len(tail_sel)} unexpected <select> block(s). FULL_PSSG format should NOT have <select>.")
            error_type = error_type or "UnexpectedSelect"
        
        if len(tail_rank) == 0:
            errors.append("Missing <rerank> block in tail.")
            error_type = error_type or "MissingRerank"
        elif len(tail_rank) > 1:
            errors.append(f"Found {len(tail_rank)} <rerank> blocks in tail; expected exactly one.")
            error_type = error_type or "DuplicateRerank"

        return {
            "status_message": f"Error: {error_type or 'ValidationError'}",
            "error_type": error_type or "ValidationError",
            "errors": errors,
            "selected": [],
            "reranked": [],
            "raw": output,
        }

    rank_text = (m.group("rank") or "").strip()

    # 3) Validate <rerank>: strict chain with exactly M indices
    # Format: [i] > [j] > ... with exactly M-1 '>' separators
    chain_re = re.compile(r"^\s*\[\d+\](\s*>\s*\[\d+\]){" + str(M - 1) + r"}\s*$")
    if not chain_re.match(rank_text):
        gt_count = rank_text.count(">")
        if gt_count != (M - 1):
            errors.append(f"<rerank> must contain exactly {M-1} '>' separators, got {gt_count}.")
            error_type = error_type or "BadRerankGtCount"
        else:
            errors.append("Rerank chain must be [i] > [j] > ... with spaces only around '>'.")
            error_type = error_type or "BadRerankFormat"

    rerank_ids = _parse_indices(rank_text)

    if len(rerank_ids) != M:
        errors.append(f"<rerank> must list exactly {M} indices, got {len(rerank_ids)}.")
        error_type = error_type or "BadRerankCount"

    # 4) Range and uniqueness checks
    if rerank_ids and not all(1 <= i <= N for i in rerank_ids):
        errors.append("Some <rerank> indices are out of range [1..N].")
        error_type = error_type or "RerankOutOfRange"

    if rerank_ids and len(set(rerank_ids)) != len(rerank_ids):
        errors.append("<rerank> indices must be unique.")
        error_type = error_type or "RerankDuplicate"

    # 5) Early return on errors
    if errors:
        return {
            "status_message": f"Error: {error_type or 'ValidationError'}",
            "error_type": error_type or "ValidationError",
            "errors": errors,
            "selected": [],
            "reranked": [],
            "raw": output,
        }

    # 6) id_to_pid mapping
    missing = [i for i in rerank_ids if i not in id_to_pid]
    if missing:
        return {
            "status_message": "Error: MissingIdMap",
            "error_type": "MissingIdMap",
            "errors": [f"id_to_pid missing keys for indices: {missing}"],
            "selected": [],
            "reranked": [],
            "raw": output,
        }

    # 7) Success: For FULL_PSSG, selected = reranked (implicit selection)
    return {
        "status_message": "Success.",
        "error_type": None,
        "errors": [],
        "selected": rerank_ids,  # Implicit selection = the indices in rerank
        "reranked": [id_to_pid[i] for i in rerank_ids],
        "raw": output,
    }
# Model type to validation function mapping
# Used by reranker_rewards.py to dispatch validation based on model_type
MODEL_TYPE_TO_VALIDATION = {
    "qwen2507_distilled_instruct_thinking_rerank": validate_qwen2507_instruct_thinking_output,
    "direct_rerank": validate_qwen2507_direct_rerank_output
}
