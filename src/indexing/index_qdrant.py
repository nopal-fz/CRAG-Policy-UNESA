import argparse, json
from typing import List, Dict, Any
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

COLLECTION = "unesa_pedoman"

def load_chunks(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path, "r", encoding="utf-8")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="data/chunks.jsonl")
    ap.add_argument("--qdrant_url", default="http://localhost:6333")
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-small")
    args = ap.parse_args()

    client = QdrantClient(url=args.qdrant_url)
    embedder = SentenceTransformer(args.embed_model)

    chunks = load_chunks(args.chunks)
    dim = embedder.get_sentence_embedding_dimension()

    # recreate collection (dev-friendly)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )

    points = []
    for i, c in enumerate(tqdm(chunks, desc="Embedding chunks")):
        text = c["text"]
        vec = embedder.encode("passage: " + text, normalize_embeddings=True).tolist()
        payload = {
            "chunk_id": c["chunk_id"],
            "bab": c.get("bab",""),
            "section": c.get("section",""),
            "subsection": c.get("subsection",""),
            "page_start": c.get("page_start"),
            "page_end": c.get("page_end"),
            "text": text,  # simpan text untuk display + BM25
        }
        points.append(qm.PointStruct(id=i, vector=vec, payload=payload))

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Indexed {len(points)} chunks into Qdrant collection='{COLLECTION}'")

if __name__ == "__main__":
    main()