import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


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


def load_registry(path: Path = BUNDLED_REGISTRY) -> list[ModelInfo]:
    with open(path) as f:
        data = yaml.safe_load(f)

    models = []
    for m in data["models"]:
        req = m["requirements"]
        perf = m.get("performance", {})
        sources = m.get("sources", {})

        models.append(ModelInfo(
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
        ))

    return models


def filter_by_tag(models: list[ModelInfo], tag: str) -> list[ModelInfo]:
    tag = tag.lower().strip()
    return [m for m in models if tag in m.tags]


def search_by_name(models: list[ModelInfo], query: str) -> list[ModelInfo]:
    """Fuzzy search models by name, id, or family."""
    from rapidfuzz import fuzz, process

    query = query.lower().strip()

    # Build a map of search string → model
    candidates = {}
    for m in models:
        candidates[m.id] = m
        candidates[m.family] = m
        candidates[m.name.lower()] = m

    results = process.extract(
        query,
        list(candidates.keys()),
        scorer=fuzz.partial_ratio,
        limit=10,
        score_cutoff=50,
    )

    # Deduplicate while preserving order
    seen = set()
    matched = []
    for match_str, score, _ in results:
        model = candidates[match_str]
        if model.id not in seen:
            seen.add(model.id)
            matched.append(model)

    return matched