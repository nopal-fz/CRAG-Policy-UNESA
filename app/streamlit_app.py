import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import streamlit as st

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.retrieval.hybrid_retriever import build_bm25
from src.retrieval.reranker import Reranker
from src.retrieval.query_transform import QueryTransformer
from src.retrieval.crag import crag_retrieve
from src.generation.ollama_generate import OllamaAnswerer


st.set_page_config(
    page_title="UNESA CRAG (Ollama)", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS tanpa sidebar
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Global Styling */
    .main {
        background: linear-gradient(135deg, #f0f4f8 0%, #e6eef5 100%);
        font-family: 'Inter', sans-serif;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Fix untuk semua text */
    p, span, div, label {
        color: #1e293b !important;
    }
    
    /* Header Title */
    h1 {
        color: #1e40af !important;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
    }
    
    /* Caption/Subtitle */
    .stCaption {
        text-align: center;
        color: #64748b !important;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Settings Card */
    .settings-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }
    
    /* Text Input */
    .stTextInput input {
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 1rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        background: white;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
        color: #1e293b !important;
    }
    
    .stTextInput input:focus {
        border-color: #2563eb;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.25);
        transform: translateY(-2px);
    }
    
    .stTextInput input::placeholder {
        color: #94a3b8 !important;
    }
    
    .stTextInput label {
        color: #1e293b !important;
        font-weight: 600;
    }
    
    /* Slider */
    .stSlider {
        padding: 0.5rem 0;
    }
    
    .stSlider label {
        color: #1e293b !important;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #1e293b !important;
        font-weight: 600;
    }
    
    /* Button */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        width: 100%;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    /* Subheader */
    h2 {
        color: #1e40af !important;
        font-weight: 700;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #2563eb !important;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Info Box */
    .stInfo {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stInfo p {
        color: #1e40af !important;
    }
    
    /* Warning Box */
    .stWarning {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stWarning p {
        color: #92400e !important;
    }
    
    /* Markdown containers */
    .stMarkdown {
        color: #1e293b !important;
    }
    
    /* Text Display */
    .stText {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #1e293b !important;
    }
    
    /* JSON Display */
    .stJson {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Columns */
    [data-testid="column"] {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin: 0.5rem;
    }
    
    [data-testid="column"] * {
        color: #1e293b !important;
    }
    
    /* Block Container */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 100%;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f1f5f9;
        border-radius: 8px;
        font-weight: 600;
        color: #1e40af !important;
    }
    
    /* Custom divider */
    hr {
        border: none;
        border-top: 2px solid #e2e8f0;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_components(ollama_model: str):
    client = QdrantClient(url="http://localhost:6333")
    embedder = SentenceTransformer("intfloat/multilingual-e5-small")
    reranker = Reranker("BAAI/bge-reranker-base", device=None)

    chunks = [json.loads(l) for l in open("data/chunks.jsonl", "r", encoding="utf-8")]
    chunks_payload = [{
        "chunk_id": c["chunk_id"],
        "text": c["text"],
        "bab": c.get("bab",""),
        "section": c.get("section",""),
        "subsection": c.get("subsection",""),
        "page_start": c.get("page_start"),
        "page_end": c.get("page_end"),
    } for c in chunks]

    bm25 = build_bm25(chunks_payload)

    qt = QueryTransformer(ollama_model=ollama_model, temperature=0.0)
    answerer = OllamaAnswerer(model=ollama_model, temperature=0.1)

    return client, embedder, reranker, chunks_payload, bm25, qt, answerer


def rujukan_str(p):
    sec = (p.get("section","") or "").strip()
    bab = (p.get("bab","") or "").strip()
    pages = f"hlm {p.get('page_start')}-{p.get('page_end')}"
    if bab and sec:
        return f"{bab} – {sec} ({pages})"
    if bab:
        return f"{bab} ({pages})"
    return f"({pages})"


# Header
st.title("UNESA Pedoman Akademik with CRAG")
st.caption("Query Transform → Hybrid Search → Reranking → CRAG Gate → Answer Generation")

# Settings in expander
with st.expander("Pengaturan", expanded=False):
    col_set1, col_set2, col_set3 = st.columns(3)
    
    with col_set1:
        ollama_model = st.text_input("Model Ollama", value="qwen2.5:7b-instruct")
    
    with col_set2:
        min_rerank = st.slider("Min Rerank Score", 0.0, 1.0, 0.10, 0.01)
        min_cov = st.slider("Min Coverage", 0.0, 1.0, 0.05, 0.01)
    
    with col_set3:
        show_debug = st.checkbox("Tampilkan Debug Info", value=False)

# Load components
client, embedder, reranker, chunks_payload, bm25, qt, answerer = load_components(ollama_model)

st.markdown("---")

# Input section
st.markdown("### Tanyakan Sesuatu")
question = st.text_input(
    "Pertanyaan Anda:", 
    placeholder="Contoh: Bagaimana prosedur cuti akademik?",
    label_visibility="collapsed"
)

if st.button("Dapatkan Jawaban"):
    if not question:
        st.warning("Silakan masukkan pertanyaan terlebih dahulu!")
    else:
        with st.spinner("Memproses pertanyaan Anda..."):
            top, debug = crag_retrieve(
                question=question,
                client=client,
                embedder=embedder,
                chunks_payload=chunks_payload,
                bm25=bm25,
                reranker=reranker,
                qt=qt,
                min_rerank=min_rerank,
                min_cov=min_cov,
            )

        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("## Jawaban")
            ans = answerer.answer(question, top)
            st.markdown(f"""
            <div style='background: white; padding: 2rem; border-radius: 12px; 
                        box-shadow: 0 2px 8px rgba(0,0,0,0.08); 
                        border-left: 4px solid #3b82f6; color: #1e293b;
                        line-height: 1.8;'>
                {ans}
            </div>
            """, unsafe_allow_html=True)

            if show_debug:
                st.markdown("---")
                st.markdown("## Debug Info")
                st.json(debug)

        with col2:
            st.markdown("## Sumber Evidence")
            if not top:
                st.info("Tidak ada evidence yang lolos gate.")
            else:
                for i, item in enumerate(top, start=1):
                    p = item["payload"]
                    
                    # Evidence card
                    st.markdown(f"""
                    <div style='background: white; padding: 1.5rem; border-radius: 12px; 
                                margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                                border-left: 4px solid #3b82f6;'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
                            <h4 style='color: #1e40af; margin: 0; font-weight: 600;'>#{i} {p.get('chunk_id')}</h4>
                            <div>
                                <span style='display: inline-block; background: #3b82f6; color: white; 
                                             padding: 0.25rem 0.75rem; border-radius: 20px; 
                                             font-size: 0.85rem; font-weight: 600; margin: 0 0.25rem;'>
                                    Hybrid: {item.get('score_hybrid',0):.4f}
                                </span>
                                <span style='display: inline-block; background: #2563eb; color: white; 
                                             padding: 0.25rem 0.75rem; border-radius: 20px; 
                                             font-size: 0.85rem; font-weight: 600; margin: 0 0.25rem;'>
                                    Rerank: {item.get('score_rerank',0):.4f}
                                </span>
                            </div>
                        </div>
                        <p style='color: #64748b; font-size: 0.9rem; margin-bottom: 0;'>
                            {rujukan_str(p)}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    txt = p.get("text","")
                    st.text(txt[:1400] + ("..." if len(txt) > 1400 else ""))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; margin-top: 2rem; padding: 1.5rem; 
            background: rgba(255,255,255,0.7); border-radius: 12px;'>
    <p style='color: #64748b; margin: 0; font-size: 0.95rem;'>
        Powered by <strong style='color: #1e40af;'>UNESA CRAG</strong> | Query Transform + Hybrid Retrieval + Ollama
    </p>
</div>
""", unsafe_allow_html=True)