"""
Free LLM Chatbot - FastAPI backend (v2)

Features:
- Streaming chat (Server-Sent Events) with token-by-token replies
- Tool calling: web search, calculator, clock, RAG over uploaded documents
- Multi-conversation sidebar with SQLite persistence (survives restarts)
- Per-conversation system prompts, temperature control, model picker
- Local Ollama backend by default; Hugging Face free tier as an alternative
"""

import json
import os

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

import config
import db
import llm
import rag

app = FastAPI(title="Free LLM Chatbot", version="2.0.0")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_env = Environment(loader=FileSystemLoader(os.path.join(_BASE_DIR, "templates")))


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    model: str | None = None
    conversation_id: str | None = None
    stream: bool = True


class CreateConversationRequest(BaseModel):
    system_prompt: str | None = None


class UpdateRequest(BaseModel):
    value: str = Field(..., min_length=1)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return template_env.get_template("index.html").render()


@app.get("/health")
async def health():
    if config.USE_HUGGINGFACE:
        return {"status": "ok", "backend": "huggingface", "model": config.HF_MODEL}
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            if r.status_code != 200:
                return {"status": "degraded", "backend": "ollama", "detail": "unreachable"}
            return {
                "status": "ok",
                "backend": "ollama",
                "model": config.OLLAMA_MODEL,
                "models": [m["name"] for m in r.json().get("models", [])],
            }
    except Exception as exc:
        return {"status": "degraded", "backend": "ollama", "detail": str(exc)}


@app.get("/api/models")
async def list_models():
    """Return available Ollama models plus frontend defaults."""
    if config.USE_HUGGINGFACE:
        return {"backend": "huggingface", "models": [config.HF_MODEL], "default": config.HF_MODEL}
    try:
        models = await llm.fetch_ollama_models()
    except Exception as exc:
        return {
            "backend": "ollama",
            "models": [config.OLLAMA_MODEL],
            "default": config.OLLAMA_MODEL,
            "detail": str(exc),
        }
    if config.OLLAMA_MODEL not in models and models:
        return {
            "backend": "ollama",
            "models": models,
            "default": models[0],
            "configured": config.OLLAMA_MODEL,
        }
    return {
        "backend": "ollama",
        "models": models or [config.OLLAMA_MODEL],
        "default": config.OLLAMA_MODEL,
    }


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------
@app.get("/api/conversations")
async def list_conversations():
    return [dict(row) for row in db.list_conversations()]


@app.post("/api/conversations")
async def create_conversation(body: CreateConversationRequest):
    conv = db.create_conversation(body.system_prompt or config.DEFAULT_SYSTEM_PROMPT)
    return dict(conv)


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conv = db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    data = dict(conv)
    data["messages"] = [dict(m) for m in db.get_messages(conv_id)]
    return data


@app.post("/api/conversations/{conv_id}/rename")
async def rename_conversation(conv_id: str, body: UpdateRequest):
    if not db.get_conversation(conv_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.rename_conversation(conv_id, body.value)
    return {"ok": True}


@app.post("/api/conversations/{conv_id}/system_prompt")
async def update_system_prompt(conv_id: str, body: UpdateRequest):
    if not db.get_conversation(conv_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.set_system_prompt(conv_id, body.value)
    return {"ok": True}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    db.delete_conversation(conv_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Documents (RAG)
# ---------------------------------------------------------------------------
@app.get("/api/documents")
async def list_documents():
    return [dict(row) for row in db.list_documents()]


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if config.USE_HUGGINGFACE:
        raise HTTPException(status_code=400, detail="RAG requires the local Ollama backend")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    text = rag.extract_text(file.filename, data)
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No extractable text found (supported: txt, md, csv, json, log, pdf)",
        )
    chunks = rag.chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no chunks")
    try:
        embeddings = await rag.embed_texts(chunks, config.EMBED_MODEL)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding failed — is Ollama running with '{config.EMBED_MODEL}' pulled? ({exc})",
        )
    doc_id = db.add_document(file.filename)
    db.add_chunks(doc_id, chunks, embeddings)
    return {"id": doc_id, "name": file.filename, "chunks": len(chunks)}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int):
    db.delete_document(doc_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
def _build_messages(conv: dict, user_message: str) -> list[dict]:
    history = db.get_messages(conv["id"])[-config.MAX_HISTORY_MESSAGES:]
    messages = [
        {"role": "system", "content": conv["system_prompt"] or config.DEFAULT_SYSTEM_PROMPT}
    ]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": user_message})
    return messages


@app.post("/api/chat")
async def chat(request_body: ChatRequest, request: Request):
    conv = db.get_conversation(request_body.conversation_id) if request_body.conversation_id else None
    if not conv:
        conv = db.create_conversation(config.DEFAULT_SYSTEM_PROMPT)
    conv = db.get_conversation(conv["id"])

    messages = _build_messages(conv, request_body.message)
    model = request_body.model or config.OLLAMA_MODEL
    first_user_message = db.count_user_messages(conv["id"]) == 0

    if config.USE_HUGGINGFACE or not request_body.stream:
        return await _chat_once(conv, messages, request_body, model)

    async def event_stream():
        reply = ""
        error = None
        yield _sse({"type": "start", "conversation_id": conv["id"]})
        try:
            async for event in llm.run_chat(messages, request_body.temperature, model):
                if event["type"] == "done":
                    reply = event["reply"]
                if event["type"] == "error":
                    error = event["detail"]
                if await request.is_disconnected():
                    break
                yield _sse(event)
        finally:
            db.add_message(conv["id"], "user", request_body.message, model)
            if reply:
                db.add_message(conv["id"], "assistant", reply, model)
            if first_user_message and reply:
                db.rename_conversation(conv["id"], request_body.message[:60] or "New chat")

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


async def _chat_once(conv, messages, request_body, model):
    """Non-streaming path (used for stream=false and for the HF backend)."""
    try:
        if config.USE_HUGGINGFACE:
            reply = await llm.call_huggingface(messages, request_body.temperature)
            model_used = config.HF_MODEL
        else:
            reply, meta = await llm.chat_plain(messages, request_body.temperature, model)
            model_used = meta.get("model_used", model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    db.add_message(conv["id"], "user", request_body.message, model_used)
    if reply:
        db.add_message(conv["id"], "assistant", reply, model_used)
    if db.count_user_messages(conv["id"]) == 1 and reply:
        db.rename_conversation(conv["id"], request_body.message[:60] or "New chat")

    return {
        "reply": reply,
        "model_used": model_used,
        "conversation_id": conv["id"],
        "history": [dict(m) for m in db.get_messages(conv["id"])],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)