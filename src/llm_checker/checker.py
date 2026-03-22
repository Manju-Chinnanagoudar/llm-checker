from dataclasses import dataclass
from llm_checker.hardware import HardwareProfile
from llm_checker.registry import ModelInfo


@dataclass
class CompatibilityResult:
    model: ModelInfo
    status: str          # "compatible" | "degraded" | "incompatible"
    score: int           # 0-100, used for ranking
    reasons: list[str]   # why degraded or incompatible
    warnings: list[str]  # non-blocking concerns


def check_compatibility(
    model: ModelInfo,
    hw: HardwareProfile,
) -> CompatibilityResult:
    reasons = []
    warnings = []
    score = 100

    req = model.requirements

    # ── Hard blockers (incompatible) ──────────────────────────

    if hw.total_ram_gb < req.min_ram_gb:
        reasons.append(
            f"Needs {req.min_ram_gb}GB RAM, you have {hw.total_ram_gb}GB"
        )

    if req.requires_avx and not hw.has_avx:
        reasons.append("Requires AVX — your CPU does not support it")

    if req.requires_avx2 and not hw.has_avx2:
        reasons.append("Requires AVX2 — your CPU does not support it")

    if hw.free_disk_gb < req.min_disk_gb:
        reasons.append(
            f"Needs {req.min_disk_gb}GB disk space, only {hw.free_disk_gb}GB free"
        )

    # GPU check — only block if model explicitly requires VRAM
    has_gpu = len(hw.gpus) > 0
    total_vram = sum(g.vram_gb for g in hw.gpus) if has_gpu else 0.0

    if req.min_vram_gb > 0 and total_vram < req.min_vram_gb:
        reasons.append(
            f"Needs {req.min_vram_gb}GB VRAM minimum, you have {total_vram}GB"
        )

    if reasons:
        return CompatibilityResult(
            model=model,
            status="incompatible",
            score=0,
            reasons=reasons,
            warnings=warnings,
        )

    # ── Soft warnings (degraded) ──────────────────────────────

    # Available RAM is low
    if hw.available_ram_gb < req.min_ram_gb:
        warnings.append(
            f"Low available RAM ({hw.available_ram_gb}GB free) — close other apps for best performance"
        )
        score -= 20

    # No GPU but model benefits from one
    if not has_gpu and req.recommended_vram_gb > 0:
        warnings.append(
            f"No GPU detected — running on CPU only (~{model.cpu_tokens_per_sec} tok/s)"
        )
        score -= 15

    # GPU VRAM below recommended
    if has_gpu and req.recommended_vram_gb > 0 and total_vram < req.recommended_vram_gb:
        warnings.append(
            f"VRAM below recommended ({total_vram}GB vs {req.recommended_vram_gb}GB) — expect slower inference"
        )
        score -= 10

    status = "degraded" if warnings else "compatible"

    return CompatibilityResult(
        model=model,
        status=status,
        score=score,
        reasons=reasons,
        warnings=warnings,
    )


def check_all(
    models: list[ModelInfo],
    hw: HardwareProfile,
) -> list[CompatibilityResult]:
    results = [check_compatibility(m, hw) for m in models]
    # Sort: compatible first, then degraded, then incompatible — within each by score desc
    order = {"compatible": 0, "degraded": 1, "incompatible": 2}
    return sorted(results, key=lambda r: (order[r.status], -r.score))