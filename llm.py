"""LLM backend: Ollama (local, streaming, tool calling) and Hugging Face (free API)."""

import json
import time

import httpx

import config
from tools import TOOL_DEFINITIONS, execute_tool

# ---------------------------------------------------------------------------
# Ollama models
# ---------------------------------------------------------------------------
def _canonical(name: str) -> str:
    return name[: -len(":latest")] if name.endswith(":latest") else name


async def fetch_ollama_models() -> list[str]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
        resp.raise_for_status()
        names = [_canonical(m["name"]) for m in resp.json().get("models", [])]
        return sorted(set(names), key=str.lower)


async def fetch_ollama_embedding_models() -> list[str]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
        resp.raise_for_status()
        names = [_canonical(m["name"]) for m in resp.json().get("models", [])]
        return sorted(n for n in names if "embed" in n.lower())


# ---------------------------------------------------------------------------
# Streaming chat with tool-calling loop
# ---------------------------------------------------------------------------
async def run_chat(messages: list[dict], temperature: float, model: str):
    """
    Async generator yielding event dicts:
      {"type": "token", "content": ...}   streamed text delta
      {"type": "tool",  "name": ...}      tool invocation started
      {"type": "meta", ...}               usage counters / latency
      {"type": "done", "reply": ...}
      {"type": "error", "detail": ...}
    """
    tools = list(TOOL_DEFINITIONS)
    loop = [*messages]
    prompt_tokens = eval_tokens = 0
    total_tokens = 0
    rounds = 0
    retried_without_tools = False
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            while True:
                payload = {
                    "model": model,
                    "messages": loop,
                    "stream": True,
                    "options": {"temperature": temperature},
                }
                if tools:
                    payload["tools"] = tools

                full = ""
                tool_calls = []
                async with client.stream(
                    "POST", f"{config.OLLAMA_BASE_URL}/api/chat", json=payload
                ) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        if tools and not retried_without_tools and "tool" in body.lower():
                            tools = []
                            retried_without_tools = True
                            continue
                        raise RuntimeError(
                            f"Ollama error {resp.status_code}: {body[:300]}"
                        )
                    async for raw in resp.aiter_lines():
                        if not raw:
                            continue
                        if raw.startswith("data:"):
                            raw = raw[5:]
                        try:
                            chunk = json.loads(raw)
                        except ValueError:
                            continue
                        msg = chunk.get("message") or {}
                        content = msg.get("content")
                        if content:
                            full += content
                            yield {"type": "token", "content": content}
                        for call in msg.get("tool_calls") or []:
                            fn = call.get("function") or {}
                            if fn:
                                tool_calls.append(fn)
                        if chunk.get("done"):
                            prompt_tokens = chunk.get("prompt_eval_count", prompt_tokens)
                            eval_tokens = chunk.get("eval_count", eval_tokens)

                if tool_calls and rounds < config.MAX_TOOL_ROUNDS:
                    rounds += 1
                    loop.append(
                        {"role": "assistant", "content": full, "tool_calls": tool_calls}
                    )
                    for call in tool_calls:
                        name = call.get("name", "")
                        arguments = call.get("arguments") or {}
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except ValueError:
                                arguments = {}
                        yield {"type": "tool", "name": name}
                        result = await execute_tool(name, arguments)
                        loop.append({"role": "tool", "content": str(result)})
                    continue
                break

        latency_ms = int((time.perf_counter() - started) * 1000)
        yield {
            "type": "meta",
            "model_used": model,
            "prompt_tokens": prompt_tokens,
            "eval_tokens": eval_tokens,
            "total_tokens": prompt_tokens + eval_tokens,
            "latency_ms": latency_ms,
        }
        yield {"type": "done", "reply": full}
    except Exception as exc:
        yield {"type": "error", "detail": str(exc)}


async def chat_plain(messages: list[dict], temperature: float, model: str) -> tuple[str, dict]:
    """Run run_chat without streaming and return (reply, meta)."""
    reply = ""
    meta = {}
    async for event in run_chat(messages, temperature, model):
        if event["type"] == "done":
            reply = event["reply"]
        elif event["type"] == "meta":
            meta = event
        elif event["type"] == "error":
            raise RuntimeError(event["detail"])
    return reply, meta


# ---------------------------------------------------------------------------
# Hugging Face inference (free tier, non-streaming)
# ---------------------------------------------------------------------------
async def call_huggingface(messages: list[dict], temperature: float) -> str:
    if not config.HF_TOKEN:
        raise RuntimeError(
            "Hugging Face is enabled but HF_TOKEN is not set. Add HF_TOKEN to your environment."
        )
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"https://api-inference.huggingface.co/models/{config.HF_MODEL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.HF_TOKEN}"},
            json={"messages": messages, "max_tokens": 512, "temperature": temperature},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Hugging Face error {resp.status_code}: {resp.text[:300]}")
        choices = resp.json().get("choices", [])
        if not choices:
            raise RuntimeError("Empty response from Hugging Face")
        return (choices[0].get("message") or {}).get("content", "").strip()