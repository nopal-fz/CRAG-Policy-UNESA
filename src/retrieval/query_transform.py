from __future__ import annotations
from typing import List, Dict
import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def _clean(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _parse_numbered_list(text: str) -> List[str]:
    # Ambil baris bernomor: "1. ...", "2) ...", "- ..."
    lines = []
    for ln in text.splitlines():
        ln2 = ln.strip()
        if not ln2:
            continue
        if re.match(r"^(\d+[\.\)]|\-)\s+", ln2):
            ln2 = re.sub(r"^(\d+[\.\)]|\-)\s+", "", ln2).strip()
            if ln2:
                lines.append(ln2)
    # fallback: split by ";"
    if not lines:
        parts = [p.strip() for p in re.split(r"[;\n]", text) if p.strip()]
        lines = parts[:4]
    return lines


class QueryTransformer:
    """
    Query Transformations ala referensi:
    - Query rewriting: bikin query lebih spesifik
    - Step-back prompting: bikin query lebih general untuk background
    - Sub-query decomposition: pecah jadi 2-4 sub pertanyaan
    """
    def __init__(
        self,
        ollama_model: str = "qwen2.5:7b-instruct",
        base_url: str | None = None,
        temperature: float = 0.0,
    ):
        self.llm = ChatOllama(
            model=ollama_model,
            base_url=base_url,  # None -> default Ollama local
            temperature=temperature,
        )

        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Kamu asisten yang mereformulasi query agar lebih spesifik untuk pencarian dokumen pedoman akademik."),
            ("human",
             "Original query: {q}\n"
             "Rewrite query agar lebih spesifik, formal, dan mudah match dengan istilah dokumen. "
             "Balas hanya 1 kalimat query.")
        ])

        self.stepback_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Kamu asisten yang membuat step-back query (lebih umum) untuk ambil konteks dasar dari dokumen kebijakan."),
            ("human",
             "Original query: {q}\n"
             "Buat 1 step-back query yang lebih umum. Balas hanya 1 kalimat query.")
        ])

        self.decompose_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Kamu asisten yang memecah pertanyaan kompleks jadi sub-queries untuk retrieval."),
            ("human",
             "Original query: {q}\n"
             "Pecah menjadi 2-4 sub-queries singkat dalam Bahasa Indonesia. "
             "Output format:\n"
             "1. ...\n2. ...\n3. ... (opsional)\n4. ... (opsional)")
        ])

    def transform(self, question: str, max_variants: int = 6) -> List[str]:
        q0 = _clean(question)

        # 1) rewrite
        rewrite = _clean((self.rewrite_prompt | self.llm).invoke({"q": q0}).content)

        # 2) step-back
        stepback = _clean((self.stepback_prompt | self.llm).invoke({"q": q0}).content)

        # 3) decompose
        decomp_raw = (self.decompose_prompt | self.llm).invoke({"q": q0}).content
        subqs = [_clean(x) for x in _parse_numbered_list(decomp_raw)][:4]

        # Gabung + dedupe
        variants = [q0, rewrite, stepback] + subqs
        out = []
        for v in variants:
            v = _clean(v)
            if v and v not in out:
                out.append(v)
        return out[:max_variants]