"""Backend tests. The LLM calls (Ollama) are monkeypatched so no local server is needed."""

import json

import pytest

import llm

TEST_SYSTEM_PROMPT = "You are a test assistant."


async def _fake_run_chat(messages, temperature, model):
    user_msgs = [m for m in messages if m["role"] == "user"]
    last = user_msgs[-1]["content"] if user_msgs else ""
    yield {"type": "token", "content": "Hello! You said: "}
    yield {"type": "meta", "model_used": model, "prompt_tokens": 10, "eval_tokens": 5, "total_tokens": 15, "latency_ms": 42}
    yield {"type": "done", "reply": f"Hello! You said: {last}"}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(llm, "run_chat", _fake_run_chat)
    monkeypatch.setattr(llm, "fetch_ollama_models", _fake_models)


async def _fake_models():
    return ["mistral", "llama3"]


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "ollama"
    assert body["status"] in ("ok", "degraded")


def test_models_endpoint(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "mistral" in body["models"]
    assert body["default"] == "mistral"


def test_conversation_crud(client):
    created = client.post("/api/conversations", json={"system_prompt": TEST_SYSTEM_PROMPT})
    assert created.status_code == 200
    conv = created.json()
    assert conv["title"] == "New chat"
    assert conv["system_prompt"] == TEST_SYSTEM_PROMPT

    cid = conv["id"]
    listed = client.get("/api/conversations").json()
    assert any(c["id"] == cid for c in listed)

    renamed = client.post(f"/api/conversations/{cid}/rename", json={"value": "My chat"})
    assert renamed.status_code == 200

    detail = client.get(f"/api/conversations/{cid}").json()
    assert detail["title"] == "My chat"
    assert detail["messages"] == []

    updated = client.post(f"/api/conversations/{cid}/system_prompt", json={"value": "New persona"})
    assert updated.status_code == 200
    assert client.get(f"/api/conversations/{cid}").json()["system_prompt"] == "New persona"

    deleted = client.delete(f"/api/conversations/{cid}")
    assert deleted.status_code == 200
    assert client.get(f"/api/conversations/{cid}").status_code == 404


def test_chat_non_stream(client):
    conv = client.post("/api/conversations", json={}).json()
    resp = client.post(
        "/api/chat",
        json={
            "message": "hello",
            "temperature": 0.7,
            "conversation_id": conv["id"],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"].startswith("Hello! You said: hello")
    assert body["model_used"] == "mistral"
    assert body["conversation_id"] == conv["id"]
    assert len(body["history"]) == 2

    # title auto-set from the first message
    detail = client.get(f"/api/conversations/{conv['id']}").json()
    assert detail["title"] == "hello"


def test_chat_stream_sse(client):
    conv = client.post("/api/conversations", json={}).json()
    resp = client.post(
        "/api/chat",
        json={"message": "hi", "conversation_id": conv["id"], "stream": True},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = resp.text
    assert '"type": "start"' in text
    assert '"type": "token"' in text
    assert '"type": "done"' in text
    assert "conversation_id" in text


def test_chat_creates_conversation_automatically(client):
    resp = client.post("/api/chat", json={"message": "first", "stream": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"]
    assert client.get(f"/api/conversations/{body['conversation_id']}").status_code == 200


def test_chat_rejects_empty_message(client):
    resp = client.post("/api/chat", json={"message": "", "stream": False})
    assert resp.status_code == 422


def test_upload_document(client, monkeypatch):
    import db as db_module
    import rag as rag_module

    def fake_embed(texts, model):
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def fake_embed_async(texts, model):
        return fake_embed(texts, model)

    monkeypatch.setattr(rag_module, "embed_texts", fake_embed_async)

    resp = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", b"Python is a programming language. FastAPI is a web framework. Ollama runs models locally.", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks"] >= 1

    docs = client.get("/api/documents").json()
    assert any(d["id"] == body["id"] for d in docs)

    deleted = client.delete(f"/api/documents/{body['id']}")
    assert deleted.status_code == 200
    assert client.get("/api/documents").json() == []


def test_tools_calculator():
    from tools import calculator

    assert calculator("2 + 3 * 4") == "14"
    assert calculator("sqrt(144)") == "12"
    assert "Could not evaluate" in calculator("__import__('os').system('rm -rf /')")
    assert "Could not evaluate" in calculator("1 +")


def test_tools_clock():
    from tools import get_current_time

    out = get_current_time()
    assert "," in out and ":" in out