from typing import Any, Dict, List, Optional
import re

THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.S)
REASON_BLOCK_RE = re.compile(r"<reason>(.*?)</reason>", re.S)
SEL_BLOCK_RE = re.compile(r"<select>(.*?)</select>", re.S)
RERANK_BLOCK_RE = re.compile(r"<rerank>(.*?)</rerank>", re.S)
IDX_RE = re.compile(r"\[(\d+)\]")


def _parse_indices(text: str) -> List[int]:
    """Extract all indices in [n] format from text"""
    return [int(x) for x in IDX_RE.findall(text)]

def validate_rerank_output(
    output: str, N: int, M: int, docid_map: Dict[str, str]
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
            "reranked": [],
            "reranked_docs": [],
            "raw": output,
        }

    # 6) id_to_pid mapping
    missing = [i for i in rerank_ids if i not in docid_map]
    if missing:
        return {
            "status_message": "Error: MissingIdMap",
            "error_type": "MissingIdMap",
            "errors": [f"id_to_pid missing keys for indices: {missing}"],
            "reranked": [],
            "reranked_docs": [],
            "raw": output,
        }

    # 7) Success: For FULL_PSSG, selected = reranked (implicit selection)
    return {
        "status_message": "Success.",
        "error_type": None,
        "errors": [],
        "reranked": rerank_ids,
        "reranked_docs": [docid_map[i] for i in rerank_ids],
        "raw": output,
    }