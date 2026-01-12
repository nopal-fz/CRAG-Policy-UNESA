from __future__ import annotations
from typing import List, Dict, Any, Tuple

from src.retrieval.hybrid_retriever import dense_search, bm25_search, merge_hybrid
from src.utils.text_utils import content_keywords
from src.retrieval.reranker import Reranker
from src.retrieval.query_transform import QueryTransformer

def keyword_coverage(query: str, text: str) -> float:
    kws = content_keywords(query)
    if not kws:
        return 0.0
    t = text.lower()
    hit = sum(1 for k in kws if k in t)
    return hit / len(kws)

def evidence_good(
    question: str,
    reranked: List[Dict[str, Any]],
    min_rerank: float = 0.1,
    min_cov: float = 0.25,
) -> Tuple[bool, Dict[str, float]]:
    if not reranked:
        return False, {"rerank_top": 0.0, "coverage": 0.0}

    top_r = reranked[0].get("score_rerank", 0.0)
    joined = "\n".join([c["payload"]["text"] for c in reranked])
    cov = keyword_coverage(question, joined)

    ok = (top_r >= min_rerank) and (cov >= min_cov)
    return ok, {"rerank_top": float(top_r), "coverage": float(cov)}

def crag_retrieve(
    question: str,
    client,
    embedder,
    chunks_payload,
    bm25,
    reranker: Reranker,
    qt: QueryTransformer,
    k_dense: int = 20,
    k_lex: int = 20,
    k_pool: int = 30,
    k_final: int = 6,
    min_rerank: float = 0.1,
    min_cov: float = 0.25,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    variants = qt.transform(question, max_variants=6)
    debug: Dict[str, Any] = {"variants": variants, "attempts": []}

    # Try a few variants (normal)
    for v in variants[:3]:
        dense = dense_search(client, embedder, v, topk=k_dense)
        lex = bm25_search(bm25, chunks_payload, v, topk=k_lex)
        pool = merge_hybrid(dense, lex, topk=k_pool)
        top = reranker.rerank(question, pool, topk=k_final)

        ok, metrics = evidence_good(question, top, min_rerank=min_rerank, min_cov=min_cov)
        debug["attempts"].append({
            "variant": v,
            "pool_size": len(pool),
            "top_chunk_ids": [t["chunk_id"] for t in top],
            **metrics,
            "ok": ok,
        })
        if ok:
            return top, debug

    # Corrective: bigger k on best variant
    v = variants[0] if variants else question
    dense = dense_search(client, embedder, v, topk=max(40, k_dense))
    lex = bm25_search(bm25, chunks_payload, v, topk=max(40, k_lex))
    pool = merge_hybrid(dense, lex, topk=max(60, k_pool))
    top = reranker.rerank(question, pool, topk=k_final)

    ok, metrics = evidence_good(question, top, min_rerank=min_rerank, min_cov=min_cov)
    debug["attempts"].append({
        "variant": v + " (corrective: bigger k)",
        "pool_size": len(pool),
        "top_chunk_ids": [t["chunk_id"] for t in top],
        **metrics,
        "ok": ok,
    })

    if ok:
        return top, debug

    return [], debug