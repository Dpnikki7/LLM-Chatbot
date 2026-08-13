# Free LLM Chatbot

A complete, modular chatbot using **Python + FastAPI** and a **free LLM**
(Ollama locally or Hugging Face). No OpenAI, ChatGPT, Gemini, or paid APIs.

## Features

| Area | Feature |
|------|--------|
| Chat UX | **Token-by-token streaming replies** (Server-Sent Events) |
| | **Markdown rendering + syntax-highlighted code blocks** |
| | **Cinematic neon-aurora UI**: animated particle background, glassmorphism, glow effects, streaming shimmer |
| | **Stop generation** button, **copy message** button, typing indicator |
| | **Temperature control** slider (0–2) |
| Memory | **Multiple conversations** in a sidebar — create / rename / delete |
| | **SQLite persistence** (`chatbot.db`) — history survives server restarts |
| | Last 20 messages kept as context (configurable) |
| Prompting | **System prompt editor** per conversation (persona/rules) |
| Models | **Model picker dropdown** — lists every installed Ollama model |
| | Hugging Face free tier as an alternative backend |
| Tools | **Web search** (free DuckDuckGo, no API key) |
| | **Calculator** (safe, sandboxed expression evaluation) |
| | **Clock** (current date/time) |
| RAG | **Upload documents** (txt/md/csv/json/log/pdf) — drag & drop or button |
| | Semantic search over your documents (Ollama embeddings) |
| Voice | **Voice input** (Web Speech API) and **voice replies** |
| Ops | **Docker** one-command deployment, **pytest** test suite |

The model decides automatically when to call a tool (e.g. "what\'s 2+2?",
"search the web for…", "summarize my uploaded PDF"). No special syntax needed.

## Project structure

```
.
├── app.py               # FastAPI routes (chat SSE, conversations, documents)
├── config.py            # Environment-based configuration
├── db.py                # SQLite persistence layer
├── llm.py               # Ollama streaming + tool-call loop, Hugging Face client
├── rag.py               # Document chunking, embeddings, semantic retrieval
├── tools.py             # Tool definitions + implementations (web search, math…)
├── templates/
│   └── index.html       # Chat UI (sidebar, streaming, markdown, voice…)
├── tests/               # pytest suite (Ollama calls mocked — no network needed)
├── requirements.txt     # Runtime dependencies
├── requirements-dev.txt # Test dependencies
├── Dockerfile           # Container build
├── docker-compose.yml   # App + Ollama together
└── README.md
```

## Quick start (Windows)

```powershell
cd "LLM-Chatbot"

# 1. Create a virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Install Ollama (https://ollama.com), start it, and pull a model
#    (a model with tool support is recommended: llama3.1, qwen2.5, mistral)
ollama run llama3.1
#    for document Q&A (RAG) also pull an embedding model once:
ollama pull nomic-embed-text

# 3. Run the app
python app.py
# open http://localhost:8000
```

macOS/Linux: replace `venv\\Scripts\\activate` with `source venv/bin/activate`.

## Optional environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `mistral` | Default chat model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for RAG |
| `MAX_HISTORY_MESSAGES` | `20` | Context window (message count) |
| `MAX_TOOL_ROUNDS` | `4` | Max tool-call loops per reply |
| `USE_HF` | `false` | Set `true` to use Hugging Face instead of Ollama |
| `HF_TOKEN` | — | Hugging Face access token (needed when `USE_HF=true`) |
| `HF_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Hugging Face model |

## Using the Hugging Face backend (no local model)

```powershell
set USE_HF=true
set HF_TOKEN=your_free_token_from_huggingface.co
python app.py
```

> Tools and RAG require the local Ollama backend; the HF backend offers plain chat.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Chat UI |
| GET | `/health` | Backend health + installed models |
| GET | `/api/models` | Installed Ollama models + default |
| GET | `/api/conversations` | List conversations |
| POST | `/api/conversations` | Create (optional `system_prompt`) |
| GET | `/api/conversations/{id}` | Conversation + full message history |
| POST | `/api/conversations/{id}/rename` | Rename |
| POST | `/api/conversations/{id}/system_prompt` | Set system prompt |
| DELETE | `/api/conversations/{id}` | Delete conversation |
| GET | `/api/documents` | List indexed documents |
| POST | `/api/documents/upload` | Upload a file (multipart `file`) |
| DELETE | `/api/documents/{id}` | Remove a document |
| POST | `/api/chat` | Send a message (SSE stream or JSON) |

`POST /api/chat` body: `{ "message", "temperature"?, "model"?, "conversation_id"?, "stream"? }`

When `stream: true` the response is Server-Sent Events:
`start` → `token`* → `tool`* → `meta` → `done` (or `error`).
When `stream: false` (or on the HF backend) a plain JSON reply is returned.

## Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The suite mocks the LLM, so it runs fully offline.

## Docker

```bash
docker compose up --build
# app: http://localhost:8000   Ollama: http://localhost:11434
# pull a model once inside the container:
docker exec -it ollama ollama run llama3.1
docker exec ollama ollama pull nomic-embed-text
```

## Troubleshooting

- **"Ollama error 404 / model not found"** — pull the model first:
  `ollama run <name>` once, or select an installed model from the dropdown.
- **"Embedding failed — is Ollama running with 'nomic-embed-text' pulled?"** —
  run `ollama pull nomic-embed-text`.
- **Tools never trigger** — models like `llama3.1`/`qwen2.5`/`mistral` (v0.3+)
  support tool calling; the app falls back to plain chat automatically for
  older models.
- **Web search returns nothing** — DuckDuckGo rate-limits occasionally; retry
  in a moment.

## Security notes

- The calculator tool only allows numeric expressions and a small whitelist of
  `math` functions (AST-validated, no `__builtins__`).
- All chat replies are sanitized (DOMPurify) before rendering as HTML.

## 🎯 About this project

This is a comprehensive, open-source chatbot project built with:
- **Modern Web Technologies**: FastAPI + Jinja2 + Web Speech API
- **Free and Open Source LLMs**: Ollama (local) + Hugging Face (free tier)
- **Production Ready**: Docker deployment, testing, multi-conversation support
- **Rich Features**: Token streaming, Markdown, tools, RAG, voice input, and more

This is version 2.0 of the project, featuring an upgraded UI with cinematic
neon-aurora aesthetics, enhanced streaming capabilities, and improved
multi-conversation management.

## License

MIT