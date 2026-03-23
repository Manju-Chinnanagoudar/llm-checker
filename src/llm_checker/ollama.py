import os
import httpx

# respect OLLAMA_HOST if set, otherwise fall back to default
# users can set this like: OLLAMA_HOST=http://localhost:11435 llm-checker list
_default_host = "http://localhost:11434"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", _default_host).rstrip("/")


def get_installed_models(host: str = OLLAMA_HOST) -> set[str]:
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=3)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return {m["name"] for m in models}
    except Exception:
        return set()


def ollama_is_running(host: str = OLLAMA_HOST) -> bool:
    try:
        httpx.get(host, timeout=2)
        return True
    except Exception:
        return False
