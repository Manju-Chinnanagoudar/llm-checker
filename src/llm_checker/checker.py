from dataclasses import dataclass, field

from llm_checker.hardware import HardwareProfile
from llm_checker.registry import ModelInfo


@dataclass
class CompatibilityResult:
    model: ModelInfo
    status: str       # compatible | degraded | incompatible
    score: int        # 0-100
    reasons: list[str] = field(default_factory=list)   # hard blockers
    warnings: list[str] = field(default_factory=list)  # soft issues


def check_compatibility(model: ModelInfo, hw: HardwareProfile) -> CompatibilityResult:
    req = model.requirements
    reasons = []
    warnings = []
    score = 100

    total_vram = sum(g.vram_gb for g in hw.gpus) if hw.gpus else 0.0

    # hard blockers — any of these means it simply won't run
    if hw.total_ram_gb < req.min_ram_gb:
        reasons.append(f"Needs {req.min_ram_gb} GB RAM, you have {hw.total_ram_gb} GB")

    if req.requires_avx and not hw.has_avx:
        reasons.append("Requires AVX — not supported by your CPU")

    if req.requires_avx2 and not hw.has_avx2:
        reasons.append("Requires AVX2 — not supported by your CPU")

    if hw.free_disk_gb < req.min_disk_gb:
        reasons.append(f"Needs {req.min_disk_gb} GB disk, only {hw.free_disk_gb} GB free")

    if req.min_vram_gb > 0 and total_vram < req.min_vram_gb:
        reasons.append(f"Needs {req.min_vram_gb} GB VRAM minimum, you have {total_vram} GB")

    if reasons:
        return CompatibilityResult(model=model, status="incompatible", score=0, reasons=reasons)

    # soft warnings — will run but not great
    if hw.available_ram_gb < req.min_ram_gb:
        warnings.append(
            f"Only {hw.available_ram_gb} GB RAM free — close other apps before running"
        )
        score -= 20

    if not hw.gpus and req.recommended_vram_gb > 0:
        warnings.append(f"CPU-only mode, expect ~{model.cpu_tokens_per_sec} tok/s")
        score -= 15
    elif hw.gpus and 0 < total_vram < req.recommended_vram_gb:
        warnings.append(
            f"VRAM below recommended ({total_vram} GB vs {req.recommended_vram_gb} GB)"
        )
        score -= 10

    status = "degraded" if warnings else "compatible"
    return CompatibilityResult(model=model, status=status, score=score, warnings=warnings)


def check_all(models: list[ModelInfo], hw: HardwareProfile) -> list[CompatibilityResult]:
    results = [check_compatibility(m, hw) for m in models]
    rank = {"compatible": 0, "degraded": 1, "incompatible": 2}
    return sorted(results, key=lambda r: (rank[r.status], -r.score))