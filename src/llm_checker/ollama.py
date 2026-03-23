import httpx

OLLAMA_URL = "http://localhost:11434"


def get_installed_models() -> set[str]:
    # returns empty set if ollama isn't running — that's fine
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return {m["name"] for m in models}
    except Exception:
        return set()


def ollama_is_running() -> bool:
    try:
        httpx.get(f"{OLLAMA_URL}", timeout=2)
        return True
    except Exception:
        return False
