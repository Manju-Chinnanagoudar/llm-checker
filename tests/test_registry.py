import pytest
from llm_checker.registry import load_registry, filter_by_tag, search_by_name


@pytest.fixture
def models():
    return load_registry()


def test_loads_models(models):
    assert len(models) > 0


def test_all_models_have_required_fields(models):
    for m in models:
        assert m.id, f"missing id: {m}"
        assert m.name, f"missing name: {m}"
        assert isinstance(m.tags, list)
        assert m.requirements.min_ram_gb >= 0


def test_filter_by_tag_coding(models):
    coding = filter_by_tag(models, "coding")
    assert len(coding) > 0
    assert all("coding" in m.tags for m in coding)


def test_filter_by_tag_unknown_returns_empty(models):
    results = filter_by_tag(models, "doesnotexist_xyz")
    assert results == []


def test_search_finds_mistral(models):
    results = search_by_name(models, "mistral")
    assert any("mistral" in m.id for m in results)


def test_search_finds_qwen_with_partial(models):
    # simulates the "queen -> qwen" case
    results = search_by_name(models, "qwen")
    assert len(results) > 0


def test_search_returns_empty_for_garbage(models):
    results = search_by_name(models, "zzzzzzzzznomatch")
    assert len(results) == 0
