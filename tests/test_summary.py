"""Tests for summary.py: Groq-first summarization, local Ollama fallback."""

# Reaches into the module's own prompt template and Groq wrapper on purpose.
# pylint: disable=protected-access

import time
from types import SimpleNamespace

import httpx
from groq import RateLimitError

import summary


def _fake_rate_limit_error(message="rate_limit_exceeded: OTPM"):
    """A real groq.RateLimitError, shaped like the 429 Groq actually returns."""
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/x"))
    return RateLimitError(message, response=response, body=None)


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


# A response with every section _summarize_groq requires present — the
# baseline "this is a complete answer" fixture for tests that aren't
# specifically exercising the missing-section check.
_COMPLETE_GROQ_RESPONSE = "\n\n".join(
    f"## {header}\ncontenido" for header in summary._GROQ_REQUIRED_HEADER_TEXT
)


class _SequentialFakeGroqClient:  # pylint: disable=too-few-public-methods
    """Stand-in for groq.Groq that returns one canned response per call, in
    order — for chunking, which makes several sequential requests on the
    same client. Each entry in ``responses`` is either a
    ``(content, finish_reason, completion_tokens)`` tuple, or an exception
    instance to raise instead (for exercising retry behavior). ``.calls``
    records every prompt sent, in order; ``.temperatures`` records the
    ``temperature`` kwarg for the same calls, in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.temperatures = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        """Pop the next canned response (or exception) for this call."""
        self.calls.append(kwargs["messages"][0]["content"])
        self.temperatures.append(kwargs.get("temperature"))
        entry = self._responses.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        content, finish_reason, completion_tokens = entry
        message = SimpleNamespace(content=content, reasoning=None)
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        usage = (
            SimpleNamespace(completion_tokens=completion_tokens)
            if completion_tokens is not None
            else None
        )
        return SimpleNamespace(choices=[choice], usage=usage)


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
    """_summarize_groq returns the text when the model finished normally
    and included every required section."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(
        summary, "Groq", lambda **_kw: _FakeGroqClient(_COMPLETE_GROQ_RESPONSE, "stop")
    )

    assert summary._summarize_groq("prompt") == _COMPLETE_GROQ_RESPONSE


def test_summarize_groq_raises_when_a_required_section_is_missing(monkeypatch):
    """A clean 'stop' with plenty of unused token budget is still rejected if
    the model skipped a required section — the real failure mode observed in
    production: finish_reason='stop', completion_tokens far under the cap,
    but whole sections (Decisiones, Compromisos, Pendientes) just missing."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    incomplete = "## Resumen\ntexto\n\n## Puntos clave\n- un punto"
    monkeypatch.setattr(
        summary, "Groq", lambda **_kw: _FakeGroqClient(incomplete, "stop", 510),
    )

    try:
        summary._summarize_groq("prompt")
        raise AssertionError("expected a RuntimeError for a response missing sections")
    except RuntimeError as e:
        assert "truncat" in str(e)
        assert "decisiones" in str(e)


def test_missing_headers_tolerates_case_punctuation_and_heading_level():
    """A model that varies case, adds a trailing colon, or uses a different
    heading level for an otherwise-compliant response isn't wrongly flagged
    as having skipped the section — only content that's genuinely absent is."""
    varied = "# Resumen\ntexto\n\n## Puntos Clave\n- x\n\n## Decisiones:\n- y\n\n"
    varied += "###   compromisos   \n- z\n\n## Pendientes\n- w"

    assert summary._missing_headers(varied) == []


def test_missing_headers_still_flags_a_genuinely_absent_section():
    """Tolerance for formatting doesn't mean it stops catching a real gap —
    a response with no 'pendientes' heading anywhere is still flagged."""
    content = (
        "## Resumen\ntexto\n\n## Puntos clave\n- x\n\n## Decisiones\n- y\n\n## Compromisos\n- z"
    )

    assert summary._missing_headers(content) == ["pendientes"]


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
    """A short but genuinely complete answer (low completion_tokens, every
    required section present) is not penalized just for being short."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(
        summary, "Groq",
        lambda **_kw: _FakeGroqClient(_COMPLETE_GROQ_RESPONSE, "stop", 50),
    )

    assert summary._summarize_groq("prompt") == _COMPLETE_GROQ_RESPONSE


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


def test_groq_response_missing_a_section_falls_back_to_local(monkeypatch, read_log):
    """generate_summary falls back to local when Groq's answer is missing a
    required section, even with tokens to spare and a clean 'stop' — the
    actual failure mode seen in production (log showed finish_reason='stop',
    completion_tokens=510/8192, yet Decisiones/Compromisos/Pendientes were
    all missing from the delivered summary)."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    incomplete = "## Resumen\ntexto\n\n## Puntos clave\n- un punto"
    monkeypatch.setattr(
        summary, "Groq", lambda **_kw: _FakeGroqClient(incomplete, "stop", 510),
    )
    monkeypatch.setattr(summary, "_summarize_local", lambda _prompt: "resumen local completo")

    text, backend = summary.generate_summary("texto de la reunion")

    assert text == "resumen local completo"
    assert backend == f"local · Ollama {summary.QWEN_MODEL}"
    assert "truncat" in read_log()


def test_detailed_prompt_template_carries_the_transcript_and_the_spanish_instruction():
    """Sanity check on the detailed prompt (local Ollama only): the Spanish
    requirement is stated twice — up front and again in the format section —
    after a real run came back in English despite the original single
    mention (qwen3:1.7b ignored it for a long, dense transcript)."""
    prompt = summary._PROMPT_TEMPLATE.format(text="HOLA_MUNDO_UNICO")
    assert "HOLA_MUNDO_UNICO" in prompt
    assert "tu respuesta completa debe estar SIEMPRE en español" in prompt
    assert "Usa Markdown claro, profesional y en español" in prompt
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


# ---------------------------------------------------------------------------
# Chunking: a transcript longer than GROQ_CHUNK_CHARS is split, summarized
# chunk by chunk, then synthesized into one final structured summary.
# ---------------------------------------------------------------------------

def test_split_into_chunks_short_text_is_one_chunk():
    """Text under the limit comes back as a single chunk, unchanged."""
    assert summary._split_into_chunks("hola", max_chars=100) == ["hola"]


def test_split_into_chunks_respects_max_chars_and_keeps_all_content():
    """A long, newline-rich text splits into within-limit chunks, losing nothing."""
    text = "linea de prueba\n" * 500  # ~8.5k chars, newline every ~16
    chunks = summary._split_into_chunks(text, max_chars=2000)

    assert len(chunks) > 1
    assert all(len(c) <= 2000 for c in chunks)
    assert "\n".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_split_into_chunks_hard_cuts_a_line_with_no_newline():
    """With no newline to break on, the splitter falls back to hard character cuts."""
    text = "x" * 9000
    chunks = summary._split_into_chunks(text, max_chars=4000)

    assert [len(c) for c in chunks] == [4000, 4000, 1000]


def test_split_into_chunks_prefers_a_blank_line_over_a_later_single_newline():
    """A blank line (a paragraph or speaker-change boundary, when the
    transcript has one) within range is preferred as the cut point over a
    single newline that sits closer to max_chars — keeping a turn intact
    matters more than using every available character."""
    text = "AAAA\n\nBBBB\nCCCC\n\nDDDD"
    chunks = summary._split_into_chunks(text, max_chars=16)

    assert chunks[0] == "AAAA"
    assert "\n".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_split_into_chunks_repeats_overlap_chars_between_consecutive_chunks():
    """Each chunk after the first repeats the tail of the previous one, so a
    point made right at a cut — a sentence, a decision — lands in both
    chunks instead of being lost to whichever side didn't get it."""
    text = "a" * 15 + "\n" + "b" * 15 + "\n" + "c" * 15  # 47 chars, cuts at 15/15/15
    chunks = summary._split_into_chunks(text, max_chars=20, overlap_chars=5)

    assert chunks == [
        "a" * 15,
        "a" * 5 + "b" * 15,
        "b" * 5 + "c" * 15,
    ]


def test_summarize_groq_chunked_splits_summarizes_and_synthesizes(monkeypatch):
    """A long transcript is split into chunks, each summarized on its own,
    then combined into one final structured summary via a synthesis call."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(summary, "GROQ_CHUNK_CHARS", 20)
    text = "a" * 15 + "\n" + "b" * 15 + "\n" + "c" * 15  # -> 3 chunks at max_chars=20

    fake = _SequentialFakeGroqClient([
        ("PUNTOS: nota uno", "stop", 50),
        ("PUNTOS: nota dos", "stop", 50),
        ("PUNTOS: nota tres", "stop", 50),
        (_COMPLETE_GROQ_RESPONSE, "stop", 300),
    ])
    monkeypatch.setattr(summary, "Groq", lambda **_kw: fake)

    result = summary._summarize_groq_chunked(text)

    assert result == _COMPLETE_GROQ_RESPONSE
    assert len(fake.calls) == 4  # 3 chunks + 1 synthesis
    synthesis_prompt = fake.calls[-1]
    assert "nota uno" in synthesis_prompt
    assert "nota dos" in synthesis_prompt
    assert "nota tres" in synthesis_prompt


def test_summarize_groq_chunked_raises_and_stops_early_if_a_chunk_fails(monkeypatch):
    """A single failing chunk aborts the whole attempt — the synthesis call
    never runs, so nothing partial is ever combined or returned."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(summary, "GROQ_CHUNK_CHARS", 20)
    text = "a" * 15 + "\n" + "b" * 15

    fake = _SequentialFakeGroqClient([
        ("PUNTOS: nota uno", "stop", 50),
        ("cortado", "length", 500),
    ])
    monkeypatch.setattr(summary, "Groq", lambda **_kw: fake)

    try:
        summary._summarize_groq_chunked(text)
        raise AssertionError("expected a RuntimeError when a chunk fails")
    except RuntimeError as e:
        assert "truncat" in str(e)
    assert len(fake.calls) == 2  # synthesis never got called


def test_summarize_groq_chunked_retries_a_chunk_that_hits_a_transient_rate_limit(
    monkeypatch,
):
    """A chunk rejected once for OTPM succeeds on retry — the whole attempt
    isn't aborted just because one small request hit a transient rate limit."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(summary, "GROQ_CHUNK_CHARS", 20)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    text = "a" * 15  # -> 1 chunk

    fake = _SequentialFakeGroqClient([
        _fake_rate_limit_error(),          # chunk 1, attempt 1: rejected
        ("PUNTOS: nota uno", "stop", 50),  # chunk 1, attempt 2: succeeds
        (_COMPLETE_GROQ_RESPONSE, "stop", 300),  # synthesis
    ])
    monkeypatch.setattr(summary, "Groq", lambda **_kw: fake)

    result = summary._summarize_groq_chunked(text)

    assert result == _COMPLETE_GROQ_RESPONSE
    assert len(fake.calls) == 3


def test_summarize_groq_chunked_aborts_once_a_chunk_exhausts_its_retries(monkeypatch):
    """A chunk that keeps hitting the rate limit past its retry budget still
    aborts the whole attempt — retrying isn't unlimited."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(summary, "GROQ_CHUNK_CHARS", 20)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    text = "a" * 15  # -> 1 chunk
    attempts = 1 + len(summary._CHUNK_RETRY_DELAYS_SECONDS)

    fake = _SequentialFakeGroqClient([_fake_rate_limit_error() for _ in range(attempts)])
    monkeypatch.setattr(summary, "Groq", lambda **_kw: fake)

    try:
        summary._summarize_groq_chunked(text)
        raise AssertionError("expected a RateLimitError once retries are exhausted")
    except RateLimitError:
        pass
    assert len(fake.calls) == attempts  # no call left over for a synthesis that never ran


def test_summarize_groq_chunked_does_not_retry_a_non_rate_limit_failure(monkeypatch):
    """A chunk failing for any reason other than a rate limit (network error,
    truncation, ...) is not retried — waiting wouldn't fix any of those."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(summary, "GROQ_CHUNK_CHARS", 20)
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)
    text = "a" * 15  # -> 1 chunk

    fake = _SequentialFakeGroqClient([RuntimeError("connection reset")])
    monkeypatch.setattr(summary, "Groq", lambda **_kw: fake)

    try:
        summary._summarize_groq_chunked(text)
        raise AssertionError("expected the RuntimeError to propagate immediately")
    except RuntimeError as e:
        assert "connection reset" in str(e)
    assert len(fake.calls) == 1  # no retry attempted
    assert not slept  # and no backoff wait happened either


def test_summarize_groq_chunked_uses_a_lower_temperature_for_the_synthesis_call(monkeypatch):
    """Per-chunk note extraction and the final synthesis call use different,
    intentional temperatures — synthesis is deterministic merging of notes
    already extracted, not fresh generation from the raw transcript."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(summary, "GROQ_CHUNK_CHARS", 20)
    text = "a" * 15 + "\n" + "b" * 15  # -> 2 chunks

    fake = _SequentialFakeGroqClient([
        ("PUNTOS: nota uno", "stop", 50),
        ("PUNTOS: nota dos", "stop", 50),
        (_COMPLETE_GROQ_RESPONSE, "stop", 300),
    ])
    monkeypatch.setattr(summary, "Groq", lambda **_kw: fake)

    summary._summarize_groq_chunked(text)

    assert fake.temperatures[:2] == [summary._GROQ_TEMPERATURE, summary._GROQ_TEMPERATURE]
    assert fake.temperatures[2] == summary._GROQ_SYNTHESIS_TEMPERATURE


def test_generate_summary_retries_chunked_after_an_otpm_rate_limit(monkeypatch, read_log):
    """A single-shot request rejected for being too large for OTPM
    (RateLimitError) is retried chunked instead of falling back right away —
    the real failure mode: a transcript well under GROQ_CHUNK_CHARS can still
    get rejected, since Groq's pre-flight estimate scales with content, not
    just length, so a fixed size threshold can't reliably predict it."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")

    def raise_rate_limit(_prompt):
        raise _fake_rate_limit_error()

    monkeypatch.setattr(summary, "_summarize_groq", raise_rate_limit)
    monkeypatch.setattr(summary, "_summarize_groq_chunked", lambda _text: "resumen chunked")

    text, backend = summary.generate_summary("texto corto que igual excede OTPM")

    assert text == "resumen chunked"
    assert backend == f"Groq · {summary.GROQ_MODEL}"
    assert "OTPM" in read_log()


def test_generate_summary_does_not_chunk_for_a_non_rate_limit_failure(monkeypatch):
    """A Groq failure that ISN'T an OTPM rejection (network error, missing
    sections, ...) falls straight back to local — chunking wouldn't fix any
    of those, so it's never attempted."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")

    def raise_missing_sections(_prompt):
        raise RuntimeError("summary response truncated: missing section(s) ## Pendientes")

    monkeypatch.setattr(summary, "_summarize_groq", raise_missing_sections)

    def boom(_text):
        raise AssertionError("chunking must not run for a non-rate-limit failure")

    monkeypatch.setattr(summary, "_summarize_groq_chunked", boom)
    monkeypatch.setattr(summary, "_summarize_local", lambda _prompt: "resumen local")

    text, backend = summary.generate_summary("texto")

    assert text == "resumen local"
    assert backend == f"local · Ollama {summary.QWEN_MODEL}"


def test_chunked_retry_failure_falls_back_to_local_with_the_full_transcript(monkeypatch, read_log):
    """If the chunked retry ALSO fails, local gets summarized from the FULL
    original transcript — never a mix of partial Groq notes and local."""
    monkeypatch.setattr(summary, "GROQ_API_KEY", "fake-key")
    full_text = "x" * 30

    def raise_rate_limit(_prompt):
        raise _fake_rate_limit_error()

    monkeypatch.setattr(summary, "_summarize_groq", raise_rate_limit)

    def boom(_text):
        raise RuntimeError("chunk 1/6 response truncated: missing section(s) ...")

    monkeypatch.setattr(summary, "_summarize_groq_chunked", boom)
    seen = {}

    def fake_summarize_local(prompt):
        seen["prompt"] = prompt
        return "resumen local completo"

    monkeypatch.setattr(summary, "_summarize_local", fake_summarize_local)

    text, backend = summary.generate_summary(full_text)

    assert text == "resumen local completo"
    assert backend == f"local · Ollama {summary.QWEN_MODEL}"
    assert seen["prompt"] == summary._PROMPT_TEMPLATE.format(text=full_text)
    assert "truncat" in read_log()
