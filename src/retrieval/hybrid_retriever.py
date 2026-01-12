from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.utils.text_utils import tokenize_basic

COLLECTION = "unesa_pedoman"

def build_bm25(chunks_payload: List[Dict[str, Any]]) -> BM25Okapi:
    corpus = [tokenize_basic(p["text"]) for p in chunks_payload]
    return BM25Okapi(corpus)

def dense_search(
    client: QdrantClient,
    embedder: SentenceTransformer,
    query: str,
    topk: int = 20,
) -> List[Dict[str, Any]]:
    qvec = embedder.encode("query: " + query, normalize_embeddings=True).tolist()

    res = client.query_points(
        collection_name=COLLECTION,
        query=qvec,                 # vector query
        limit=topk,
        with_payload=True,
    )

    out: List[Dict[str, Any]] = []
    for p in res.points:
        out.append({
            "chunk_id": p.payload["chunk_id"],
            "score_dense": float(p.score),
            "payload": p.payload,
        })
    return out

def bm25_search(
    bm25: BM25Okapi,
    chunks_payload: List[Dict[str, Any]],
    query: str,
    topk: int = 20,
) -> List[Dict[str, Any]]:
    qtok = tokenize_basic(query)
    scores = bm25.get_scores(qtok)
    idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
    out = []
    for i in idx_sorted:
        out.append({
            "chunk_id": chunks_payload[i]["chunk_id"],
            "score_lex": float(scores[i]),
            "payload": chunks_payload[i],
        })
    return out

def merge_hybrid(
    dense_hits: List[Dict[str, Any]],
    lex_hits: List[Dict[str, Any]],
    w_dense: float = 0.7,
    w_lex: float = 0.3,
    topk: int = 30,
) -> List[Dict[str, Any]]:
    # normalize lexical scores
    lex_max = max([h.get("score_lex", 0.0) for h in lex_hits] + [1.0])

    merged: Dict[str, Dict[str, Any]] = {}

    for h in dense_hits:
        cid = h["chunk_id"]
        merged.setdefault(cid, {
            "chunk_id": cid,
            "payload": h["payload"],
            "score_dense": 0.0,
            "score_lex": 0.0
        })
        merged[cid]["score_dense"] = max(merged[cid]["score_dense"], h.get("score_dense", 0.0))

    for h in lex_hits:
        cid = h["chunk_id"]
        merged.setdefault(cid, {
            "chunk_id": cid,
            "payload": h["payload"],
            "score_dense": 0.0,
            "score_lex": 0.0
        })
        merged[cid]["score_lex"] = max(merged[cid]["score_lex"], h.get("score_lex", 0.0) / lex_max)

    results = []
    for cid, row in merged.items():
        score = (w_dense * row["score_dense"]) + (w_lex * row["score_lex"])
        results.append({
            **row,
            "score_hybrid": float(score),
        })

    results.sort(key=lambda x: x["score_hybrid"], reverse=True)
    return results[:topk]