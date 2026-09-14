"""Tests for summary.py: Groq-first summarization, local Ollama fallback."""

# Reaches into the module's own prompt template and Groq wrapper on purpose.
# pylint: disable=protected-access

from types import SimpleNamespace

import summary


class _FakeCompletions:  # pylint: disable=too-few-public-methods
    """Stand-in for client.chat.completions with a canned response."""

    def __init__(self, content, finish_reason="stop", completion_tokens=None):
        self._content = content
        self._finish_reason = finish_reason
        self._completion_tokens = completion_tokens

    def create(self, **_kwargs):
        """Return a fake completion shaped like the real Groq SDK response."""
        message = SimpleNamespace(content=self._content, reasoning=None)
        choice = SimpleNamespace(message=message, finish_reason=self._finish_reason)
        usage = (
            SimpleNamespace(completion_tokens=self._completion_tokens)
            if self._completion_tokens is not None
            else None
        )
        return SimpleNamespace(choices=[choice], usage=usage)


class _FakeGroqClient:  # pylint: disable=too-few-public-methods
    """Stand-in for groq.Groq exposing just .chat.completions.create."""

    def __init__(self, content, finish_reason="stop", completion_tokens=None):
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(content, finish_reason, completion_tokens)
        )


def test_groq_summary_used_when_it_succeeds(monkeypatch):
    """Groq succeeds: its text is used directly, with the CONCISE prompt, and
    local Ollama never runs."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    seen = {}

    def fake_summarize_groq(prompt):
        seen["prompt"] = prompt
        return "resumen de groq"

    monkeypatch.setattr(summary, "_summarize_groq", fake_summarize_groq)

    def boom(_prompt):
        raise AssertionError("local Ollama must not run when Groq succeeds")

    monkeypatch.setattr(summary, "_summarize_local", boom)

    text, backend = summary.generate_summary("transcripcion de prueba")

    assert text == "resumen de groq"
    assert backend == f"Groq · {summary.GROQ_MODEL}"
    assert seen["prompt"] == summary._PROMPT_TEMPLATE_GROQ.format(text="transcripcion de prueba")


def test_groq_failure_falls_back_to_local_with_the_detailed_prompt(monkeypatch, read_log):
    """A Groq failure falls back to local Ollama using the ORIGINAL detailed
    prompt (not the concise one Groq got), and the reason is logged."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    seen = {}

    def raise_rate_limit(_prompt):
        raise RuntimeError("rate_limit_exceeded")

    def fake_summarize_local(prompt):
        seen["prompt"] = prompt
        return "resumen local"

    monkeypatch.setattr(summary, "_summarize_groq", raise_rate_limit)
    monkeypatch.setattr(summary, "_summarize_local", fake_summarize_local)

    text, backend = summary.generate_summary("texto de la reunion")

    assert text == "resumen local"
    assert backend == f"local · Ollama {summary.QWEN_MODEL}"
    assert seen["prompt"] == summary._PROMPT_TEMPLATE.format(text="texto de la reunion")
    log_text = read_log()
    assert "Groq summary failed" in log_text
    assert "rate_limit_exceeded" in log_text


def test_no_api_key_skips_groq_and_uses_the_detailed_prompt(monkeypatch):
    """Without GROQ_API_KEY (the sandbox default), Groq is never attempted and
    local gets the original detailed prompt."""
    assert summary.GROQ_API_KEY == ""
    seen = {}

    def boom(_prompt):
        raise AssertionError("Groq must not run without an API key")

    def fake_summarize_local(prompt):
        seen["prompt"] = prompt
        return "resumen local"

    monkeypatch.setattr(summary, "_summarize_groq", boom)
    monkeypatch.setattr(summary, "_summarize_local", fake_summarize_local)

    text, backend = summary.generate_summary("texto")

    assert text == "resumen local"
    assert backend.startswith("local ·")
    assert seen["prompt"] == summary._PROMPT_TEMPLATE.format(text="texto")


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


def test_summarize_groq_raises_when_completion_tokens_land_near_the_cap(monkeypatch):
    """A clean finish_reason='stop' is distrusted when completion_tokens lands
    right at the configured cap — qwen/qwen3.8-27b's reasoning trace can eat
    most of the shared budget and leave the visible answer short without Groq
    ever reporting "length" for it (see _GROQ_NEAR_CAP_RATIO)."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    near_cap_tokens = int(summary.GROQ_MAX_COMPLETION_TOKENS * 0.99)
    monkeypatch.setattr(
        summary, "Groq",
        lambda **_kw: _FakeGroqClient("## Resumen\ncorto", "stop", near_cap_tokens),
    )

    try:
        summary._summarize_groq("prompt")
        raise AssertionError("expected a RuntimeError for a near-cap completion")
    except RuntimeError as e:
        assert "truncat" in str(e)


def test_summarize_groq_accepts_a_short_answer_well_under_the_cap(monkeypatch):
    """A short but genuinely complete answer (low completion_tokens) is not
    penalized just for being short."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(
        summary, "Groq",
        lambda **_kw: _FakeGroqClient("## Resumen\ncorto pero completo", "stop", 50),
    )

    assert summary._summarize_groq("prompt") == "## Resumen\ncorto pero completo"


def test_truncated_groq_response_falls_back_to_local(monkeypatch, read_log):
    """generate_summary falls back to local, with the detailed prompt, when
    Groq's answer got cut short."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(
        summary, "Groq", lambda **_kw: _FakeGroqClient("resumen a medi", "length")
    )
    seen = {}

    def fake_summarize_local(prompt):
        seen["prompt"] = prompt
        return "resumen local completo"

    monkeypatch.setattr(summary, "_summarize_local", fake_summarize_local)

    text, backend = summary.generate_summary("texto de la reunion")

    assert text == "resumen local completo"
    assert backend == f"local · Ollama {summary.QWEN_MODEL}"
    assert seen["prompt"] == summary._PROMPT_TEMPLATE.format(text="texto de la reunion")
    assert "truncat" in read_log()


def test_near_cap_groq_response_falls_back_to_local_even_with_a_clean_stop(monkeypatch, read_log):
    """generate_summary falls back to local when Groq's completion_tokens lands
    at the cap, even though finish_reason itself claims a clean 'stop'."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    near_cap_tokens = int(summary.GROQ_MAX_COMPLETION_TOKENS * 0.99)
    monkeypatch.setattr(
        summary, "Groq",
        lambda **_kw: _FakeGroqClient("## Resumen\ncorto", "stop", near_cap_tokens),
    )
    monkeypatch.setattr(summary, "_summarize_local", lambda _prompt: "resumen local completo")

    text, backend = summary.generate_summary("texto de la reunion")

    assert text == "resumen local completo"
    assert backend == f"local · Ollama {summary.QWEN_MODEL}"
    assert "truncat" in read_log()


def test_detailed_prompt_template_carries_the_transcript_and_the_spanish_instruction():
    """Sanity check on the detailed prompt (local Ollama only)."""
    prompt = summary._PROMPT_TEMPLATE.format(text="HOLA_MUNDO_UNICO")
    assert "HOLA_MUNDO_UNICO" in prompt
    assert "Usa Markdown claro, profesional y español" in prompt
    assert "No inventes información" in prompt


def test_both_prompts_require_ticket_numbers_to_stay_as_one_4_digit_number():
    """Neither prompt should let the model split a ticket number like 3619
    into '36.19' or '36/19' — this was observed in a real summary."""
    for template in (summary._PROMPT_TEMPLATE, summary._PROMPT_TEMPLATE_GROQ):
        prompt = template.format(text="texto")
        assert "3619" in prompt
        assert "36.19" in prompt
        assert "36/19" in prompt


def test_groq_prompt_template_is_short_and_capped():
    """Sanity check on the concise prompt (Groq only): shorter, capped output,
    same anti-hallucination rule, and distinct from the detailed one."""
    prompt = summary._PROMPT_TEMPLATE_GROQ.format(text="HOLA_MUNDO_UNICO")
    assert "HOLA_MUNDO_UNICO" in prompt
    assert "máximo 3-4 viñetas por sección" in prompt
    assert "no lo adivines" in prompt
    assert len(summary._PROMPT_TEMPLATE_GROQ) < len(summary._PROMPT_TEMPLATE)
