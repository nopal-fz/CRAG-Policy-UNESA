# src/chunk_by_heading.py
import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, Any, List, Tuple


BAB_RE = re.compile(r"^\s*BAB\s+([IVXLCDM]+)\s*$", re.IGNORECASE)
LETTER_RE = re.compile(r"^\s*([A-Z])\.\s+(.+)$")
ROMAN_RE = re.compile(r"^\s*([IVXLCDM]+)\.\s+(.+)$")  # contoh: I. Status ...


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    # buang line kosong beruntun
    out = []
    for ln in lines:
        if not ln:
            continue
        ln = re.sub(r"\s+", " ", ln)
        out.append(ln)
    return out


def detect_repeated_lines(pages: List[Dict[str, Any]], freq_threshold: float = 0.35) -> set:
    """
    Cari baris yang muncul di banyak halaman (biasanya header/footer).
    """
    all_lines = []
    for p in pages:
        lines = set(normalize_lines(p["text"]))
        for ln in lines:
            # filter kandidat header/footer: pendek & bukan heading BAB
            if len(ln) <= 60 and not BAB_RE.match(ln):
                all_lines.append(ln)

    cnt = Counter(all_lines)
    n_pages = max(1, len(pages))
    repeated = set()
    for ln, c in cnt.items():
        if (c / n_pages) >= freq_threshold:
            repeated.add(ln)
    return repeated


def parse_heading(line: str) -> Tuple[str, str]:
    """
    Return (heading_type, heading_value) or ("", "")
    """
    m = BAB_RE.match(line)
    if m:
        return ("bab", f"BAB {m.group(1).upper()}")

    m = LETTER_RE.match(line)
    if m:
        return ("letter", f"{m.group(1)}. {m.group(2).strip()}")

    m = ROMAN_RE.match(line)
    if m and len(m.group(1)) <= 6:  # biar gak ketabrak angka/teks lain
        return ("roman", f"{m.group(1)}. {m.group(2).strip()}")

    return ("", "")


def chunk_pages(
    pages: List[Dict[str, Any]],
    max_chars: int = 3500,
) -> List[Dict[str, Any]]:
    repeated = detect_repeated_lines(pages)

    chunks = []
    curr = {
        "bab": "",
        "section": "",
        "subsection": "",
        "page_start": None,
        "page_end": None,
        "lines": [],
    }

    def flush():
        nonlocal curr, chunks
        text = "\n".join(curr["lines"]).strip()
        if not text:
            curr["lines"] = []
            return

        chunk_id = f"p{curr['page_start']}_p{curr['page_end']}_{len(chunks):05d}"
        chunks.append({
            "chunk_id": chunk_id,
            "bab": curr["bab"],
            "section": curr["section"],
            "subsection": curr["subsection"],
            "page_start": curr["page_start"],
            "page_end": curr["page_end"],
            "text": text,
        })
        curr["lines"] = []

    for p in pages:
        page_no = p["page"]
        lines = normalize_lines(p["text"])
        # remove repeated header/footer lines
        lines = [ln for ln in lines if ln not in repeated]

        for ln in lines:
            htype, hval = parse_heading(ln)

            # start new logical block on heading
            if htype:
                flush()
                if htype == "bab":
                    curr["bab"] = hval
                    curr["section"] = ""
                    curr["subsection"] = ""
                elif htype == "letter":
                    curr["section"] = hval
                    curr["subsection"] = ""
                elif htype == "roman":
                    curr["subsection"] = hval

                # set page range for new chunk
                curr["page_start"] = page_no if curr["page_start"] is None else curr["page_start"]
                curr["page_end"] = page_no
                # include heading line as part of chunk context
                curr["lines"].append(ln)
                continue

            # normal content
            if curr["page_start"] is None:
                curr["page_start"] = page_no
            curr["page_end"] = page_no
            curr["lines"].append(ln)

            # hard split by size
            if sum(len(x) for x in curr["lines"]) >= max_chars:
                flush()
                # carry over minimal context header (optional)
                carry = []
                if curr["bab"]:
                    carry.append(curr["bab"])
                if curr["section"]:
                    carry.append(curr["section"])
                if curr["subsection"]:
                    carry.append(curr["subsection"])
                curr["lines"] = carry
                curr["page_start"] = page_no
                curr["page_end"] = page_no

    flush()
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input raw_pages.jsonl")
    ap.add_argument("--out", required=True, help="Output chunks.jsonl")
    ap.add_argument("--max_chars", type=int, default=3500)
    args = ap.parse_args()

    pages = load_jsonl(args.inp)
    chunks = chunk_pages(pages, max_chars=args.max_chars)
    save_jsonl(args.out, chunks)

    print(f"Saved {len(chunks)} chunks to {args.out}")


if __name__ == "__main__":
    main()