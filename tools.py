"""Tools the LLM can call: calculator, clock, web search, and RAG document retrieval."""

import ast
import datetime
import html
import inspect
import math
import re
import urllib.parse

import httpx

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
TOP_RESULTS = 5
RESULT_MAX_CHARS = 160

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for up-to-date information and return a list of "
                "matching results with titles, URLs and snippets. Use it whenever "
                "the user asks about recent events, facts you are unsure about, "
                "or anything that may have changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression (e.g. '2 + 3 * 4', 'sqrt(144)'). Returns the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression to evaluate"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_docs",
            "description": "Search previously uploaded documents (PDF/TXT/MD/CSV) using semantic search and return the most relevant excerpts. Use it when the user asks questions about their own uploaded files.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Question or search phrase"}},
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Calculator (safe, AST-based evaluation)
# ---------------------------------------------------------------------------
_SAFE_NAMES = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
_SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})


def calculator(expression: str) -> str:
    allowed_nodes = (
        ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
        ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
        ast.USub, ast.UAdd,
    )

    def check(node):
        if not isinstance(node, allowed_nodes):
            raise ValueError("expression too complex or unsafe")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants allowed")
        if isinstance(node, ast.Name) and node.id not in _SAFE_NAMES:
            raise ValueError(f"function '{node.id}' not allowed")
        for child in ast.iter_child_nodes(node):
            check(child)

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        check(tree)
        value = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, _SAFE_NAMES)
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)
    except Exception as exc:
        return f"Could not evaluate '{expression}': {exc}"


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------
def get_current_time() -> str:
    return datetime.datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")


# ---------------------------------------------------------------------------
# Web search (free DuckDuckGo endpoints, no API key)
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_html_results(page: str) -> list[dict]:
    anchors = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.S
    )
    snippets = re.findall(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', page, re.S
    )
    results = []
    for i, (url, title) in enumerate(anchors):
        snippet = _clean(snippets[i]) if i < len(snippets) else ""
        results.append({"url": url, "title": _clean(title), "snippet": snippet})
    return results


def _parse_lite_results(page: str) -> list[dict]:
    anchors = re.findall(
        r'<a[^>]*class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.S
    )
    snippets = re.findall(r'<td class="result-snippet">(.*?)</td>', page, re.S)
    results = []
    for i, (url, title) in enumerate(anchors):
        snippet = _clean(snippets[i]) if i < len(snippets) else ""
        results.append({"url": url, "title": _clean(title), "snippet": snippet})
    return results


def _format_results(query: str, results: list[dict]) -> str:
    if not results:
        return f'No results found for "{query}".'
    lines = [f"Web search results for '{query}':"]
    for i, r in enumerate(results[:TOP_RESULTS], 1):
        title = r["title"][:RESULT_MAX_CHARS]
        snippet = r["snippet"][:RESULT_MAX_CHARS]
        lines.append(f"{i}. {title}\n   {r['url']}\n   {snippet}")
    return "\n".join(lines)


async def web_search(query: str) -> str:
    encoded = urllib.parse.quote_plus(query)
    attempts = [
        f"https://html.duckduckgo.com/html/?q={encoded}",
        f"https://lite.duckduckgo.com/lite/?q={encoded}",
    ]
    headers = {"User-Agent": USER_AGENT}
    for idx, url in enumerate(attempts):
        try:
            async with httpx.AsyncClient(
                timeout=15.0, headers=headers, follow_redirects=True
            ) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    continue
                results = _parse_html_results(resp.text) if idx == 0 else _parse_lite_results(resp.text)
                if results:
                    return _format_results(query, results)
        except Exception:
            continue
    return f'Web search failed for "{query}" (network error or rate limit).'


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------
async def execute_tool(name: str, arguments: dict) -> str:
    from rag import retrieve_docs  # local import to avoid a circular dependency

    impls = {
        "web_search": web_search,
        "calculator": calculator,
        "get_current_time": get_current_time,
        "retrieve_docs": retrieve_docs,
    }
    fn = impls.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        result = fn(**arguments) if inspect.iscoroutinefunction(fn) is False else await fn(**arguments)
        return str(result)
    except Exception as exc:
        return f"Tool '{name}' failed: {exc}"