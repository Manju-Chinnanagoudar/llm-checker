import pytest
from llm_checker.hardware import HardwareProfile, GPUInfo
from llm_checker.registry import load_registry, ModelInfo, ModelRequirements
from llm_checker.checker import check_compatibility, check_all


# a decent mid-range machine — most models should be compatible
@pytest.fixture
def decent_hw():
    return HardwareProfile(
        total_ram_gb=16.0,
        available_ram_gb=10.0,
        cpu_name="Test CPU",
        cpu_cores=8,
        has_avx=True,
        has_avx2=True,
        has_avx512=False,
        gpus=[GPUInfo(name="Test GPU", vram_gb=8.0, backend="cuda")],
        free_disk_gb=100.0,
        os="linux",
    )


@pytest.fixture
def weak_hw():
    return HardwareProfile(
        total_ram_gb=4.0,
        available_ram_gb=1.5,
        cpu_name="Old CPU",
        cpu_cores=2,
        has_avx=False,
        has_avx2=False,
        has_avx512=False,
        gpus=[],
        free_disk_gb=5.0,
        os="linux",
    )


@pytest.fixture
def small_model():
    return ModelInfo(
        id="test-1b",
        name="Test 1B",
        family="test",
        tags=["general"],
        requirements=ModelRequirements(
            min_ram_gb=2.0,
            min_vram_gb=0.0,
            recommended_vram_gb=4.0,
            requires_avx=True,
            requires_avx2=False,
            min_disk_gb=1.0,
        ),
        cpu_tokens_per_sec=20,
        gpu_tokens_per_sec=80,
        ollama_name="test:1b",
        huggingface_name="test/test-1b",
        backends=["ollama"],
        license="mit",
        verified=True,
    )


def test_compatible_on_decent_hw(small_model, decent_hw):
    result = check_compatibility(small_model, decent_hw)
    assert result.status == "compatible"
    assert result.score == 100
    assert result.reasons == []


def test_incompatible_no_avx(small_model, weak_hw):
    result = check_compatibility(small_model, weak_hw)
    assert result.status == "incompatible"
    assert any("AVX" in r for r in result.reasons)


def test_degraded_no_gpu(small_model, decent_hw):
    # strip GPU from decent hw
    hw = HardwareProfile(**{**decent_hw.__dict__, "gpus": []})
    result = check_compatibility(small_model, hw)
    assert result.status == "degraded"
    assert any("CPU" in w for w in result.warnings)


def test_check_all_sorts_correctly(decent_hw):
    models = load_registry()
    results = check_all(models, decent_hw)
    statuses = [r.status for r in results]
    # once we hit incompatible, no compatible/degraded should follow
    seen_incompatible = False
    for s in statuses:
        if s == "incompatible":
            seen_incompatible = True
        if seen_incompatible:
            assert s == "incompatible"
