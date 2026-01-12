import json
import random

path = "data/chunks.jsonl"
rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
print("chunks:", len(rows))
for r in random.sample(rows, k=min(5, len(rows))):
    print("\n---", r["chunk_id"], r["bab"], r["section"], r["subsection"], f"p{r['page_start']}-p{r['page_end']}")
    print(r["text"][:400], "...")