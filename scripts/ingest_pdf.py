# src/ingest_pdf.py
import argparse
import json
import os
from typing import Dict, Any, List
import pdfplumber


def extract_pages(pdf_path: str) -> List[Dict[str, Any]]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # normalisasi sederhana
            text = text.replace("\u00a0", " ").strip()
            pages.append({"page": idx, "text": text})
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path ke PDF")
    ap.add_argument("--out", required=True, help="Output raw_pages.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pages = extract_pages(args.pdf)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in pages:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_nonempty = sum(1 for p in pages if p["text"].strip())
    print(f"Saved {len(pages)} pages to {args.out} (non-empty: {n_nonempty})")


if __name__ == "__main__":
    main()  