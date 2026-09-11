"""Tests for summary.py: Groq-first summarization, local Ollama fallback."""

# Reaches into the module's own prompt template and Groq wrapper on purpose.
# pylint: disable=protected-access

from types import SimpleNamespace

import summary


class _FakeCompletions:  # pylint: disable=too-few-public-methods
    """Stand-in for client.chat.completions with a canned response."""

    def __init__(self, content, finish_reason="stop"):
        self._content = content
        self._finish_reason = finish_reason

    def create(self, **_kwargs):
        """Return a fake completion shaped like the real Groq SDK response."""
        message = SimpleNamespace(content=self._content, reasoning=None)
        choice = SimpleNamespace(message=message, finish_reason=self._finish_reason)
        return SimpleNamespace(choices=[choice])


class _FakeGroqClient:  # pylint: disable=too-few-public-methods
    """Stand-in for groq.Groq exposing just .chat.completions.create."""

    def __init__(self, content, finish_reason="stop"):
        self.chat = SimpleNamespace(completions=_FakeCompletions(content, finish_reason))


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


def test_summarize_groq_returns_content_on_a_normal_finish(monkeypatch):
    """_summarize_groq returns the text when the model finished normally."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(
        summary, "Groq", lambda **_kw: _FakeGroqClient("resumen completo", "stop")
    )

    assert summary._summarize_groq("prompt") == "resumen completo"


def test_summarize_groq_raises_when_truncated_by_the_token_cap(monkeypatch):
    """A finish_reason of 'length' is treated as a failure, not a partial success."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(
        summary, "Groq", lambda **_kw: _FakeGroqClient("resumen a medi", "length")
    )

    try:
        summary._summarize_groq("prompt")
        raise AssertionError("expected a RuntimeError for a truncated response")
    except RuntimeError as e:
        assert "truncat" in str(e)


def test_truncated_groq_response_falls_back_to_local(monkeypatch, read_log):
    """generate_summary falls back to local when Groq's answer got cut short."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(
        summary, "Groq", lambda **_kw: _FakeGroqClient("resumen a medi", "length")
    )
    monkeypatch.setattr(summary, "_summarize_local", lambda prompt: "resumen local completo")

    text, backend = summary.generate_summary("texto de la reunion")

    assert text == "resumen local completo"
    assert backend == f"local · Ollama {summary.QWEN_MODEL}"
    assert "truncat" in read_log()


def test_prompt_template_carries_the_transcript_and_the_spanish_instruction():
    """Sanity check: both backends share the same structured Spanish prompt."""
    prompt = summary._PROMPT_TEMPLATE.format(text="HOLA_MUNDO_UNICO")
    assert "HOLA_MUNDO_UNICO" in prompt
    assert "Usa Markdown claro, profesional y español" in prompt
    assert "No inventes información" in prompt
