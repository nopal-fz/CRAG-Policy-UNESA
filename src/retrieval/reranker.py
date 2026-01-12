from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: Optional[str] = "cuda"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], topk: int = 6) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(query, c["payload"]["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["score_rerank"] = float(s)

        candidates.sort(key=lambda x: x["score_rerank"], reverse=True)
        return candidates[:topk]