import unicodedata
from typing import List
import regex
import pytrec_eval



class Tokens:
    def __init__(self, data, annotators):
        self._data = data
        self._annotators = annotators

    def words(self, uncased=False):
        ws = [t[0] for t in self._data]
        return [w.lower() for w in ws] if uncased else ws


class SimpleTokenizer:
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self, **kwargs):
        self._regexp = regex.compile(
            f"({self.ALPHA_NUM})|({self.NON_WS})",
            flags=regex.IGNORECASE | regex.UNICODE | regex.MULTILINE,
        )
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            token = matches[i].group()
            span = matches[i].span()
            start_ws = span[0]
            end_ws = matches[i + 1].span()[0] if i + 1 < len(matches) else span[1]
            data.append((token, text[start_ws:end_ws], span))
        return Tokens(data, self.annotators)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def has_answer_string(answers: List[str], text: str, tokenizer: SimpleTokenizer) -> bool:
    """Check if any answer appears in text (case-insensitive exact match)"""
    text = _normalize(text)
    text_tok = tokenizer.tokenize(text).words(uncased=True)
    n = len(text_tok)
    for a in answers:
        a = _normalize(a)
        a_tok = tokenizer.tokenize(a).words(uncased=True)
        m = len(a_tok)
        if m == 0:
            continue
        for i in range(0, n - m + 1):
            if a_tok == text_tok[i:i+m]:
                return True
    return False


_simple_tokenizer = SimpleTokenizer()

def compute_average_hit_at_ks(answers: List[str], 
                              documents: List[dict], 
                              hit_cutoffs: List[int] = [1,3,5]) -> float:
    """
    Compute Average Hit@{k1, k2, ...}.
    
    Args:
        answers: List of ground truth answer strings
        documents: List of retrieved document dicts with 'contents' field
        hit_cutoffs: List of k values for Hit@k computation
    Returns:
        Average Hit@k score (0.0 to 1.0)
    """
    assert len(documents) <= 5, "Only support up to 5 retrieved documents"
    
    answer_hits = []
    for doc in documents: 
        content = doc.get("contents") or doc.get("text") or doc.get("passage") or ""
        if not content:
            raise ValueError("Document missing content field")

        answer_hits.append(1 if has_answer_string(answers, content, _simple_tokenizer) else 0)
        
    hit_scores = [int(sum(answer_hits[:k]) > 0) for k in hit_cutoffs]
    
    return sum(hit_scores) / len(hit_scores)

def has_answer_in_documents(answers: List[str], documents: List[dict]) -> bool:
    """Check if any answer appears in any of the documents."""
    for doc in documents:
        content = doc.get("contents") or doc.get("text") or doc.get("passage") or ""
        if not content:
            raise ValueError("Document missing content field")
        if has_answer_string(answers, content, _simple_tokenizer):
            return True
    return False


def compute_ndcg_at_m(answers: List[str],
                      all_documents: List[dict],
                      ranked_indices: List[int],
                      top_m: int = 5) -> float:
    """
    Compute NDCG@M using pytrec_eval.
    
    Relevance is binary (0/1) based on whether the document contains any answer.
    
    - qrel (relevance judgments) is built from ALL retrieved documents (top-N pool),
      so IDCG reflects the true number of relevant docs available.
    - run (system ranking) is built from ranked_indices (0-indexed positions into
      all_documents, in the order the reranker chose them).
    
    Args:
        answers: List of ground truth answer strings
        all_documents: Full retrieval pool (top-N, e.g. 50 docs) — used for qrel
        ranked_indices: 0-indexed positions into all_documents, in reranker order.
                        E.g. [2, 16, 41, 0, 27] means the reranker picked
                        all_documents[2] as #1, all_documents[16] as #2, etc.
        top_m: Cutoff for NDCG computation
    Returns:
        Tuple of (NDCG@M score (0.0 to 1.0), number of relevant docs in pool)
    """
    if not all_documents:
        return 0.0, 0

    # Build qrel from ALL docs in the pool (top-N)
    qrel = {}
    for i, doc in enumerate(all_documents):
        doc_id = f"d{i}"
        content = doc.get("contents") or doc.get("text") or doc.get("passage") or ""
        if not content:
            continue
        rel = 1 if has_answer_string(answers, content, _simple_tokenizer) else 0
        qrel[doc_id] = rel

    num_relevant = sum(v for v in qrel.values())

    if num_relevant == 0:
        # No relevant docs at all → NDCG is 0
        return 0.0, 0

    # Build run: reranker's picks get descending scores (M, M-1, ..., 1)
    run = {}
    for rank, idx in enumerate(ranked_indices):
        run[f"d{idx}"] = float(len(ranked_indices) - rank)

    evaluator = pytrec_eval.RelevanceEvaluator(
        {"q0": qrel},
        {f"ndcg_cut_{top_m}"}
    )
    results = evaluator.evaluate({"q0": run})
    return results["q0"][f"ndcg_cut_{top_m}"], num_relevant