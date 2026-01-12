import re
from typing import List

STOPWORDS_ID = {
    "apa","bagaimana","kapan","dimana","yang","dan","atau","dari","untuk","dengan","pada","di","ke","itu","ini",
    "terkait","jelaskan","sebutkan","apakah","berapa","mohon","tolong"
}

def normalize_whitespace(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_basic(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def content_keywords(query: str) -> List[str]:
    toks = tokenize_basic(query)
    toks = [t for t in toks if t not in STOPWORDS_ID and len(t) > 2]
    return toks