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


BUNDLED_REGISTRY = Path(__file__).parent / "data" / "models.yaml"


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


def load_registry(path: Path = BUNDLED_REGISTRY) -> list[ModelInfo]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [_parse_model(m) for m in data["models"]]


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
        score_cutoff=50,
    )

    seen: set[str] = set()
    results = []
    for match_str, _score, _ in hits:
        m = candidates[match_str]
        if m.id not in seen:
            seen.add(m.id)
            results.append(m)

    return results