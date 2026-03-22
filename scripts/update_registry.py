"""
Scheduled script run by GitHub Actions every 8 hours.
Merges curated models.yaml with live Ollama library data
and writes the result to registry/models.json.
"""
import json
import os
import httpx
import yaml
from pathlib import Path
from datetime import datetime, timezone

CURATED_PATH  = Path("src/llm_checker/data/models.yaml")
REGISTRY_PATH = Path("registry/models.json")


def load_curated() -> dict:
    with open(CURATED_PATH) as f:
        data = yaml.safe_load(f)
    return {m["id"]: m for m in data["models"]}


def fetch_ollama_models() -> list[dict]:
    try:
        resp = httpx.get("https://ollama.com/api/tags", timeout=15)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except Exception as e:
        print(f"⚠️  Ollama fetch failed: {e}")
        return []


def build_registry() -> list[dict]:
    curated = load_curated()
    ollama_models = fetch_ollama_models()

    registry = dict(curated)

    added = 0
    for model in ollama_models:
        mid = model.get("name", "").replace(":", "-")
        if not mid or mid in registry:
            continue
        registry[mid] = {
            "id": mid,
            "name": model.get("name", mid),
            "family": mid.split("-")[0],
            "tags": ["general"],
            "requirements": {
                "min_ram_gb": 4,
                "min_vram_gb": 0,
                "recommended_vram_gb": 0,
                "requires_avx": True,
                "requires_avx2": False,
                "min_disk_gb": 2.0,
            },
            "performance": {
                "cpu_tokens_per_sec": 10,
                "gpu_tokens_per_sec": 0,
            },
            "sources": {"ollama": model.get("name")},
            "backends": ["ollama"],
            "license": "unknown",
            "verified": False,
        }
        added += 1

    print(f"✅ {len(curated)} curated + {added} new from Ollama = {len(registry)} total")
    return list(registry.values())


if __name__ == "__main__":
    models = build_registry()
    REGISTRY_PATH.parent.mkdir(exist_ok=True)
    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(models),
        "models": models,
    }
    REGISTRY_PATH.write_text(json.dumps(output, indent=2))
    print(f"📦 Written to {REGISTRY_PATH}")
