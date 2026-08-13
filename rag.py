"""Local RAG: text extraction, chunking, embeddings (Ollama) and semantic retrieval."""

import io
import re

import httpx
import numpy as np

import config
import db

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 3
MIN_SCORE = 0.30


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from an uploaded file. Supports txt/md/csv/json/log and PDF."""
    low = filename.lower()
    if low.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks of ~CHUNK_SIZE chars at sentence boundaries."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            window = text[start + CHUNK_OVERLAP:end]
            boundary = max(window.rfind(". "), window.rfind("\n"), window.rfind("! "), window.rfind("? "))
            if boundary > 0:
                end = start + CHUNK_OVERLAP + boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
        if start >= end:
            break
    return chunks[: config.MAX_DOC_CHUNKS]


async def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    """Embed a batch of texts via Ollama /api/embed (falls back to /api/embeddings)."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{config.OLLAMA_BASE_URL}/api/embed",
            json={"model": model, "input": texts},
        )
        if resp.status_code == 404 and len(texts) == 1:
            resp = await client.post(
                f"{config.OLLAMA_BASE_URL}/api/embeddings",
                json={"model": model, "prompt": texts[0]},
            )
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding failed ({resp.status_code}): {resp.text[:200]}")
        data = resp.json()
        embeddings = data.get("embeddings") or []
        if not embeddings and "embedding" in data:
            embeddings = [data["embedding"]]
        return embeddings


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _decode_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


async def retrieve_docs(query: str, top_k: int = TOP_K) -> str:
    """Embed the query and return the most relevant uploaded-document excerpts."""
    try:
        embeddings = await embed_texts([query], config.EMBED_MODEL)
        if not embeddings:
            return "Document search produced no embedding for the query."
        query_vec = np.asarray(embeddings[0], dtype=np.float32)
        rows = db.get_all_chunks()
        if not rows:
            return "No documents uploaded yet. Tell the user to upload a file (TXT/MD/PDF/CSV) first."
        scored = []
        for row in rows:
            target = _decode_embedding(row["embedding"])
            if len(target) != len(query_vec):
                continue
            scored.append((_cosine_sim(query_vec, target), row["content"]))
        scored.sort(key=lambda item: item[0], reverse=True)
        best = [s for s in scored if s[0] >= MIN_SCORE][:top_k]
        if not best:
            return "No relevant excerpts found in the uploaded documents."
        return "\n\n---\n\n".join(
            f"[relevance {score:.2f}] {content[:800]}" for score, content in best
        )
    except Exception as exc:
        return f"Document search failed: {exc}"