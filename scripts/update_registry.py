import json
import re
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
    try:
        resp = httpx.get(
            HF_API,
            params={"filter": "gguf", "sort": "downloads", "direction": -1, "limit": 50},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"huggingface fetch failed: {e}")
        return []


def _estimate_ram(name: str) -> tuple[float, float]:
    name_lower = name.lower()
    # trillion param models first (1t = 1000b)
    match = re.search(r"(\d+\.?\d*)t\b", name_lower)
    if match:
        params_b = float(match.group(1)) * 1000
        return round(params_b * 0.6 + 1, 1), round(params_b * 0.55, 1)
    # standard billions: 7b, 671b, 1.5b etc
    match = re.search(r"(\d+\.?\d*)b\b", name_lower)
    if match:
        params_b = float(match.group(1))
        return max(round(params_b * 0.6 + 1, 1), 1.0), max(round(params_b * 0.55, 1), 0.5)
    return 4.0, 2.0


def _needs_avx2(name: str) -> bool:
    if re.search(r"\d+\.?\d*t\b", name.lower()):
        return True
    match = re.search(r"(\d+\.?\d*)b\b", name.lower())
    if match:
        return float(match.group(1)) >= 7
    return False


def _guess_tags(name: str) -> list[str]:
    n = name.lower()
    if any(x in n for x in ["code", "coder", "coding", "starcoder", "deepseek-coder"]):
        return ["coding"]
    if any(x in n for x in ["embed", "embedding"]):
        return ["embedding"]
    if any(x in n for x in ["vision", "llava", "moondream", "visual", "vl"]):
        return ["vision"]
    return ["general"]


def hf_to_registry_entry(hf_model: dict) -> dict | None:
    mid = hf_model.get("modelId", "")
    if not mid:
        return None
    slug = mid.split("/")[-1].lower().replace("_", "-")
    min_ram, min_disk = _estimate_ram(mid)
    return {
        "id": slug,
        "name": mid.split("/")[-1],
        "family": mid.split("/")[0].lower() if "/" in mid else "unknown",
        "tags": _guess_tags(mid),
        "requirements": {
            "min_ram_gb": min_ram,
            "min_vram_gb": 0,
            "recommended_vram_gb": 0,
            "requires_avx": True,
            "requires_avx2": _needs_avx2(mid),
            "min_disk_gb": min_disk,
        },
        "performance": {"cpu_tokens_per_sec": 10, "gpu_tokens_per_sec": 0},
        "sources": {"huggingface": mid},
        "backends": ["llama.cpp", "lm-studio"],
        "license": hf_model.get("license", "unknown"),
        "verified": False,
    }


def build_registry() -> list[dict]:
    curated = load_curated()
    registry = dict(curated)

    ollama_added = 0
    for m in fetch_ollama():
        mid = m.get("name", "").replace(":", "-")
        if not mid or mid in registry:
            continue
        min_ram, min_disk = _estimate_ram(mid)
        registry[mid] = {
            "id": mid,
            "name": m.get("name", mid),
            "family": mid.split("-")[0],
            "tags": _guess_tags(mid),
            "requirements": {
                "min_ram_gb": min_ram,
                "min_vram_gb": 0,
                "recommended_vram_gb": 0,
                "requires_avx": True,
                "requires_avx2": _needs_avx2(mid),
                "min_disk_gb": min_disk,
            },
            "performance": {"cpu_tokens_per_sec": 10, "gpu_tokens_per_sec": 0},
            "sources": {"ollama": m.get("name")},
            "backends": ["ollama"],
            "license": "unknown",
            "verified": False,
        }
        ollama_added += 1

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
