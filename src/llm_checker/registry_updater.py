import json
import httpx
from pathlib import Path
from datetime import datetime, timezone

REMOTE_REGISTRY_URL = (
    "https://raw.githubusercontent.com/Manju-Chinnanagoudar/llm-checker/main/registry/models.json"
)

CACHE_DIR  = Path.home() / ".llm-checker"
CACHE_FILE = CACHE_DIR / "cache.json"
META_FILE  = CACHE_DIR / "cache_meta.json"

CACHE_MAX_AGE_HOURS = 8


def cache_is_fresh() -> bool:
    if not META_FILE.exists():
        return False
    try:
        meta = json.loads(META_FILE.read_text())
        fetched_at = datetime.fromisoformat(meta["fetched_at"])
        age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
        return age_hours < CACHE_MAX_AGE_HOURS
    except Exception:
        return False


def load_cache() -> list[dict] | None:
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text())
        return data.get("models", [])
    except Exception:
        return None


def save_cache(models: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({"models": models}, indent=2))
    META_FILE.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(models),
    }))


def fetch_remote(url: str = REMOTE_REGISTRY_URL) -> list[dict] | None:
    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except Exception:
        return None


def pull_updates(force: bool = False) -> tuple[str, int]:
    """
    Returns (source, model_count) where source is one of:
    'remote_fresh', 'remote_cached', 'bundled'
    """
    if not force and cache_is_fresh():
        cached = load_cache()
        if cached:
            return ("remote_cached", len(cached))

    models = fetch_remote()

    if models:
        save_cache(models)
        return ("remote_fresh", len(models))

    # Fall back to cache even if stale
    cached = load_cache()
    if cached:
        return ("remote_cached", len(cached))

    return ("bundled", 0)
