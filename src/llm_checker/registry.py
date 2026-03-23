import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelRequirements:
    min_ram_gb: float
    min_vram_gb: float
    recommended_vram_gb: float
    requires_avx: bool
    requires_avx2: bool
    min_disk_gb: float


@dataclass
class ModelInfo:
    id: str
    name: str
    family: str
    tags: list[str]
    requirements: ModelRequirements
    cpu_tokens_per_sec: int
    gpu_tokens_per_sec: int
    ollama_name: Optional[str]
    huggingface_name: Optional[str]
    backends: list[str]
    license: str
    verified: bool
    installed: bool = False


BUNDLED_REGISTRY = Path(__file__).parent / "data" / "models.yaml"
CACHE_FILE = Path.home() / ".llm-checker" / "cache.json"


def _parse_model(m: dict) -> ModelInfo:
    req = m["requirements"]
    perf = m.get("performance", {})
    sources = m.get("sources", {})
    return ModelInfo(
        id=m["id"],
        name=m["name"],
        family=m.get("family", "unknown"),
        tags=m.get("tags", []),
        requirements=ModelRequirements(
            min_ram_gb=req.get("min_ram_gb", 0),
            min_vram_gb=req.get("min_vram_gb", 0),
            recommended_vram_gb=req.get("recommended_vram_gb", 0),
            requires_avx=req.get("requires_avx", False),
            requires_avx2=req.get("requires_avx2", False),
            min_disk_gb=req.get("min_disk_gb", 0),
        ),
        cpu_tokens_per_sec=perf.get("cpu_tokens_per_sec", 0),
        gpu_tokens_per_sec=perf.get("gpu_tokens_per_sec", 0),
        ollama_name=sources.get("ollama"),
        huggingface_name=sources.get("huggingface"),
        backends=m.get("backends", []),
        license=m.get("license", "unknown"),
        verified=m.get("verified", False),
    )


def _load_from_cache() -> list[ModelInfo] | None:
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text())
        models = [_parse_model(m) for m in data.get("models", [])]
        return models if models else None
    except Exception:
        # cache is corrupt or unreadable, fall through to bundled
        return None


def _load_from_yaml(path: Path) -> list[ModelInfo]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [_parse_model(m) for m in data["models"]]


def load_registry(path: Path = BUNDLED_REGISTRY) -> list[ModelInfo]:
    # prefer cache if available — it has more models from HF + Ollama
    cached = _load_from_cache()
    if cached:
        return cached
    # fall back to the bundled yaml (works offline, first run)
    return _load_from_yaml(path)


def get_registry_source() -> str:
    """Returns where the registry was loaded from — useful for debug/display."""
    if CACHE_FILE.exists():
        return f"cache ({CACHE_FILE})"
    return "bundled registry"


def mark_installed(models: list[ModelInfo], installed_names: set[str]) -> list[ModelInfo]:
    # match by ollama_name — e.g. "llama3.1:8b" in the installed set
    for m in models:
        if m.ollama_name and m.ollama_name in installed_names:
            m.installed = True
    return models


def filter_by_tag(models: list[ModelInfo], tag: str) -> list[ModelInfo]:
    t = tag.lower().strip()
    return [m for m in models if t in m.tags]


def search_by_name(models: list[ModelInfo], query: str) -> list[ModelInfo]:
    from rapidfuzz import fuzz, process

    q = query.lower().strip()

    # index by id, family and full name so partial matches work
    candidates: dict[str, ModelInfo] = {}
    for m in models:
        candidates[m.id] = m
        candidates[m.family] = m
        candidates[m.name.lower()] = m

    hits = process.extract(
        q,
        list(candidates.keys()),
        scorer=fuzz.partial_ratio,
        limit=10,
        score_cutoff=60,
    )

    seen: set[str] = set()
    results = []
    for match_str, _score, _ in hits:
        m = candidates[match_str]
        if m.id not in seen:
            seen.add(m.id)
            results.append(m)

    return results
