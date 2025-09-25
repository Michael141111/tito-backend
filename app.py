import os
from pathlib import Path
from typing import List, Optional
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

BASE_DIR = Path(__file__).parent
DATA_HTML_DIR = BASE_DIR / "data_html"
INDEX_DIR = BASE_DIR / "index_store"

def get_cors_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()] or ["*"]

app = FastAPI(title="Tito Backend (TF-IDF)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URL_STATIC_HTML = os.getenv("BASE_URL_STATIC_HTML", "/html")
if DATA_HTML_DIR.exists():
    app.mount(BASE_URL_STATIC_HTML, StaticFiles(directory=str(DATA_HTML_DIR)), name="html")

VEC_PATH = INDEX_DIR / "vectorizer.joblib"
MAT_PATH = INDEX_DIR / "tfidf.npz"
META_PATH = INDEX_DIR / "meta.json"

def _list_html_files():
    return sorted([p for p in DATA_HTML_DIR.glob("*.html") if p.is_file()])

def _extract_text(html_path: Path) -> str:
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

def build_index() -> int:
    files = _list_html_files()
    texts, meta = [], []
    for p in files:
        texts.append(_extract_text(p))
        meta.append({"file": p.name, "path": f"{BASE_URL_STATIC_HTML}/{p.name}"})
    if not texts:
        raise RuntimeError("No HTML files found in data_html/")

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    INDEX_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(vectorizer, VEC_PATH)

    import scipy.sparse
    scipy.sparse.save_npz(MAT_PATH, X)
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False))
    return X.shape[0]

def load_index():
    if not (VEC_PATH.exists() and MAT_PATH.exists() and META_PATH.exists()):
        return None, None, None
    vectorizer = joblib.load(VEC_PATH)
    import scipy.sparse
    X = scipy.sparse.load_npz(MAT_PATH)
    meta = json.loads(META_PATH.read_text())
    return vectorizer, X, meta

class RagQuery(BaseModel):
    query: str
    lang: Optional[str] = "en"
    k: Optional[int] = 4

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/ingest")
def ingest():
    try:
        n = build_index()
        return {"ok": True, "size": n}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/rag")
def rag(q: RagQuery):
    vectorizer, X, meta = load_index()
    if vectorizer is None:
        raise HTTPException(status_code=400, detail="Index not built yet. POST /ingest after placing HTMLs in data_html/.")
    qvec = vectorizer.transform([q.query])
    sims = cosine_similarity(qvec, X).ravel()
    top_idx = sims.argsort()[::-1][: (q.k or 4)]
    results = [{"source": meta[i]["path"], "file": meta[i]["file"], "score": float(sims[i])} for i in top_idx]
    answer = f"Top match: {meta[top_idx[0]]['file']}" if len(top_idx) else "No match."
    return {"answer": answer, "sources": results}
