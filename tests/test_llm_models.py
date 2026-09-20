from __future__ import annotations

from services.llm import FALLBACK_MODELS, PRIMARY_MODEL, llm


def test_default_chain_excludes_classifier_and_free_models():
    labels = [label for label, _ in llm()._candidates]

    assert not any("safeguard" in label for label in labels)
    assert not any("openrouter" in label for label in labels)
    assert not any(":free" in label for label in labels)


def test_fallbacks_are_tool_calling_models():
    assert PRIMARY_MODEL
    for model in FALLBACK_MODELS:
        assert "safeguard" not in model
