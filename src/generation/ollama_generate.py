from __future__ import annotations

from typing import List, Dict, Any, Optional
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Kamu adalah asisten pedoman akademik UNESA.\n"
     "TUGAS: EKSTRAK informasi dari KONTEXT, jangan menambah fakta baru.\n\n"
     "ATURAN KETAT:\n"
     "- Fokus HANYA pada topik: \"{section}\".\n"
     "- Semua langkah/ketentuan yang kamu tulis HARUS ada (tersurat) di KONTEXT.\n"
     "- Jangan membahas topik lain.\n"
     "- Output HARUS format JSON valid, TANPA teks lain.\n\n"
     "Skema JSON:\n"
     "{{\n"
     "  \"summary\": \"2-3 kalimat ringkas\",\n"
     "  \"steps\": [\"langkah 1\", \"langkah 2\", \"...\"],\n"
     "  \"notes\": [\"catatan opsional\"],\n"
     "  \"rujukan\": \"BAB – Section (hlm X–Y)\"\n"
     "}}\n\n"
     "Jika KONTEXT tidak memuat jawaban, output JSON berikut:\n"
     "{{\n"
     "  \"summary\": \"Tidak ditemukan di Pedoman Administrasi Akademik dan Kelulusan UNESA 2024.\",\n"
     "  \"steps\": [],\n"
     "  \"notes\": [],\n"
     "  \"rujukan\": \"\"\n"
     "}}"
    ),
    ("human",
     "Pertanyaan:\n{question}\n\n"
     "KONTEXT:\n{context}\n")
])


def _pick_top_payload(top_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not top_chunks:
        return {}
    top_sorted = sorted(top_chunks, key=lambda x: x.get("score_rerank", 0.0), reverse=True)
    return top_sorted[0]["payload"]


def _build_context(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""
    header = (
        f"[{payload.get('chunk_id')}] "
        f"{payload.get('bab','')} – {payload.get('section','')} "
        f"(hlm {payload.get('page_start')}-{payload.get('page_end')})"
    )
    return header + "\n" + (payload.get("text", "") or "")


def _safe_json_extract(text: str) -> str:
    # ambil blok JSON pertama
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0).strip() if m else ""


def _is_grounded(answer_text: str, context: str) -> bool:
    """
    Heuristic grounding check:
    - kalau output menyebut kata kunci yang tidak ada di context (mis. 'yudisium', 'registrasi'),
      anggap ngaco.
    """
    ctx = context.lower()
    a = answer_text.lower()

    # kata-kata “red flag” yang sering muncul saat LLM loncat topik
    red_flags = ["yudisium", "registrasi mahasiswa", "ktm", "kuota yudisium", "tefl", "toefl"]
    for rf in red_flags:
        if rf in a and rf not in ctx:
            return False

    return True


def _fallback_extractive(context: str, payload: Dict[str, Any]) -> str:
    """
    Kalau LLM ngaco, kita jawab dengan ekstraksi sederhana dari konteks:
    ambil baris yang terlihat seperti mekanisme/syarat (a., b., c., dst).
    """
    lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
    picked = []
    for ln in lines:
        if re.match(r"^[a-z]\.\s+", ln):  # a. b. c.
            picked.append(re.sub(r"^[a-z]\.\s+", "", ln).strip())
    picked = picked[:8]

    ruj = f"{payload.get('bab','')} – {payload.get('section','')} (hlm {payload.get('page_start')}-{payload.get('page_end')})"
    if not picked:
        return "Tidak ditemukan di Pedoman Administrasi Akademik dan Kelulusan UNESA 2024."

    bullets = "\n".join([f"- {p}" for p in picked])
    return (
        "Berikut mekanisme/ketentuan yang tercantum di pedoman:\n"
        f"{bullets}\n\n"
        f"Rujukan: {ruj}."
    )


class OllamaAnswerer:
    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
        self.chain = EXTRACT_PROMPT | self.llm

    def answer(self, question: str, top_chunks: List[Dict[str, Any]]) -> str:
        if not top_chunks:
            return "Tidak ditemukan di Pedoman Administrasi Akademik dan Kelulusan UNESA 2024."

        payload = _pick_top_payload(top_chunks)
        context = _build_context(payload)
        section = (payload.get("section") or "").strip() or "bagian yang relevan"
        rujukan = f"{payload.get('bab','')} – {payload.get('section','')} (hlm {payload.get('page_start')}-{payload.get('page_end')})"

        if not context.strip():
            return "Tidak ditemukan di Pedoman Administrasi Akademik dan Kelulusan UNESA 2024."

        resp = self.chain.invoke({
            "question": question,
            "context": context,
            "section": section,
        }).content

        json_blob = _safe_json_extract(resp)
        if not json_blob:
            return _fallback_extractive(context, payload)

        # grounding check: kalau ada red flag yang tidak ada di context -> fallback
        if not _is_grounded(json_blob, context):
            return _fallback_extractive(context, payload)

        summary = re.search(r"\"summary\"\s*:\s*\"(.*?)\"", json_blob, flags=re.DOTALL)
        summary_txt = summary.group(1).strip() if summary else ""

        steps = re.findall(r"\"steps\"\s*:\s*\[(.*?)\]", json_blob, flags=re.DOTALL)
        steps_block = steps[0] if steps else ""
        step_items = re.findall(r"\"(.*?)\"", steps_block, flags=re.DOTALL)

        notes = re.findall(r"\"notes\"\s*:\s*\[(.*?)\]", json_blob, flags=re.DOTALL)
        notes_block = notes[0] if notes else ""
        note_items = re.findall(r"\"(.*?)\"", notes_block, flags=re.DOTALL)

        # Render final 2–5 kalimat + bullet
        out = []
        if summary_txt:
            out.append(summary_txt)

        if step_items:
            out.append("\n**Mekanisme/Prosedur:**")
            out.extend([f"- {s}" for s in step_items[:10]])

        if note_items:
            out.append("\n**Catatan:**")
            out.extend([f"- {n}" for n in note_items[:6]])

        out.append(f"\nRujukan: {rujukan}.")
        return "\n".join(out).strip()