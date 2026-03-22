"""
Runs on a schedule via GitHub Actions (every 8 hours).
Merges the curated models.yaml with live data from Ollama and HuggingFace
and writes the result to registry/models.json.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

CURATED_PATH = Path("src/llm_checker/data/models.yaml")
REGISTRY_PATH = Path("registry/models.json")

HF_API = "https://huggingface.co/api/models"
OLLAMA_API = "https://ollama.com/api/tags"


def load_curated() -> dict:
    with open(CURATED_PATH) as f:
        raw = yaml.safe_load(f)
    return {m["id"]: m for m in raw["models"]}


def fetch_ollama() -> list[dict]:
    try:
        resp = httpx.get(OLLAMA_API, timeout=15)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except Exception as e:
        print(f"ollama fetch failed: {e}")
        return []


def fetch_hf_gguf_models() -> list[dict]:
    # pull top downloaded GGUF models — no token needed for public models
    # rate limit is 1000 req/day on the free tier, this uses 1
    try:
        resp = httpx.get(
            HF_API,
            params={
                "filter": "gguf",
                "sort": "downloads",
                "direction": -1,
                "limit": 50,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"huggingface fetch failed: {e}")
        return []


def hf_to_registry_entry(hf_model: dict) -> dict | None:
    mid = hf_model.get("modelId", "")
    if not mid:
        return None

    # use the last part of modelId as a slug
    slug = mid.split("/")[-1].lower().replace("_", "-")

    # rough tag guessing from model name — not perfect but good enough
    name_lower = mid.lower()
    tags = ["general"]
    if any(x in name_lower for x in ["code", "coder", "coding", "starcoder", "deepseek-coder"]):
        tags = ["coding"]
    elif any(x in name_lower for x in ["embed", "embedding"]):
        tags = ["embedding"]
    elif any(x in name_lower for x in ["vision", "llava", "moondream", "visual"]):
        tags = ["vision"]

    return {
        "id": slug,
        "name": mid.split("/")[-1],
        "family": mid.split("/")[0].lower() if "/" in mid else "unknown",
        "tags": tags,
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
        "sources": {"huggingface": mid},
        "backends": ["llama.cpp", "lm-studio"],
        "license": hf_model.get("license", "unknown"),
        "verified": False,
    }


def build_registry() -> list[dict]:
    curated = load_curated()
    registry = dict(curated)

    # merge ollama models
    ollama_added = 0
    for m in fetch_ollama():
        mid = m.get("name", "").replace(":", "-")
        if not mid or mid in registry:
            continue
        registry[mid] = {
            "id": mid,
            "name": m.get("name", mid),
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
            "performance": {"cpu_tokens_per_sec": 10, "gpu_tokens_per_sec": 0},
            "sources": {"ollama": m.get("name")},
            "backends": ["ollama"],
            "license": "unknown",
            "verified": False,
        }
        ollama_added += 1

    # merge huggingface GGUF models
    hf_added = 0
    for hf_m in fetch_hf_gguf_models():
        entry = hf_to_registry_entry(hf_m)
        if not entry or entry["id"] in registry:
            continue
        registry[entry["id"]] = entry
        hf_added += 1

    print(f"{len(curated)} curated + {ollama_added} ollama + {hf_added} huggingface = {len(registry)} total")
    return list(registry.values())


if __name__ == "__main__":
    models = build_registry()
    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(models),
        "models": models,
    }
    REGISTRY_PATH.parent.mkdir(exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(output, indent=2))
    print(f"written to {REGISTRY_PATH}")