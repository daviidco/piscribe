"""Tests for summary.py: Groq-first summarization, local Ollama fallback."""

# Reaches into the module's own prompt template on purpose.
# pylint: disable=protected-access

import summary


def test_groq_summary_used_when_it_succeeds(monkeypatch):
    """Groq succeeds: its text is used directly, local Ollama never runs."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(summary, "_summarize_groq", lambda prompt: "resumen de groq")

    def boom(_prompt):
        raise AssertionError("local Ollama must not run when Groq succeeds")

    monkeypatch.setattr(summary, "_summarize_local", boom)

    text, backend = summary.generate_summary("transcripcion de prueba")

    assert text == "resumen de groq"
    assert backend == f"Groq · {summary.GROQ_MODEL}"


def test_groq_failure_falls_back_to_local(monkeypatch, read_log):
    """Any Groq exception falls back to local Ollama; the reason is logged."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")

    def raise_rate_limit(_prompt):
        raise RuntimeError("rate_limit_exceeded")

    monkeypatch.setattr(summary, "_summarize_groq", raise_rate_limit)
    monkeypatch.setattr(summary, "_summarize_local", lambda prompt: "resumen local")

    text, backend = summary.generate_summary("texto de la reunion")

    assert text == "resumen local"
    assert backend == f"local · Ollama {summary.QWEN_MODEL}"
    log_text = read_log()
    assert "Groq summary failed" in log_text
    assert "rate_limit_exceeded" in log_text


def test_no_api_key_skips_groq_entirely(monkeypatch):
    """Without GROQ_API_KEY (the sandbox default) Groq is never attempted."""
    assert summary.GROQ_API_KEY == ""

    def boom(_prompt):
        raise AssertionError("Groq must not run without an API key")

    monkeypatch.setattr(summary, "_summarize_groq", boom)
    monkeypatch.setattr(summary, "_summarize_local", lambda prompt: "resumen local")

    text, backend = summary.generate_summary("texto")

    assert text == "resumen local"
    assert backend.startswith("local ·")


def test_prompt_template_carries_the_transcript_and_the_spanish_instruction():
    """Sanity check: both backends share the same structured Spanish prompt."""
    prompt = summary._PROMPT_TEMPLATE.format(text="HOLA_MUNDO_UNICO")
    assert "HOLA_MUNDO_UNICO" in prompt
    assert "Responde SIEMPRE en español" in prompt
    assert "nunca inventes ni adivines nombres" in prompt
