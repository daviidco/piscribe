"""Tests for rag.py: /ask retrieval + Groq-first/local-fallback answer drafting."""

# Reaches into the module's own private helpers and Groq wrapper, same as test_summary.py.
# pylint: disable=protected-access,too-few-public-methods

from types import SimpleNamespace

import rag


class _FakeEmbedResponse:
    """Stand-in for ollama.EmbedResponse — just the .embeddings attribute."""

    def __init__(self, vectors):
        self.embeddings = vectors


class _FakeCompletions:
    """Stand-in for client.chat.completions with a canned response."""

    def __init__(self, content):
        self._content = content

    def create(self, **_kwargs):
        """Return a fake completion shaped like the real Groq SDK response."""
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class _FakeGroqClient:
    """Stand-in for groq.Groq exposing just .chat.completions.create."""

    def __init__(self, content):
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


def _chunk(filename, text, vector, kind="transcript", created_at="2026-01-01 00:00:00"):
    return {
        "filename": filename, "kind": kind, "text": text, "embedding": vector,
        "created_at": created_at,
    }


def _fake_embed(vector):
    """An ollama.embed stand-in that always answers with ``vector``."""
    return lambda **_kwargs: _FakeEmbedResponse([vector])


def test_cosine_similarity_identical_vectors_is_one():
    """Two identical vectors point in exactly the same direction."""
    assert rag._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    """Two orthogonal vectors are considered entirely unrelated."""
    assert rag._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_handles_a_zero_vector_without_raising():
    """A chunk that somehow embedded to an all-zero vector must not raise a
    ZeroDivisionError — it's simply treated as unrelated (similarity 0)."""
    assert rag._cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0


def test_search_ranks_by_similarity_best_first_and_caps_at_top_k(monkeypatch):
    """_search returns at most RAG_TOP_K matches, best cosine similarity first."""
    monkeypatch.setattr(rag, "RAG_TOP_K", 2)
    chunks = [
        _chunk("a.mp4", "poco relacionado", [0.0, 1.0]),
        _chunk("b.mp4", "muy relacionado", [1.0, 0.0]),
        _chunk("c.mp4", "algo relacionado", [0.7, 0.7]),
    ]
    monkeypatch.setattr(rag.store, "all_chunks", lambda: chunks)

    matches = rag._search([1.0, 0.0])

    assert len(matches) == 2
    assert matches[0][0]["filename"] == "b.mp4"
    assert matches[0][1] >= matches[1][1]


def test_answer_question_below_threshold_returns_canned_answer_without_any_llm(monkeypatch):
    """Nothing crossing RAG_MIN_SIMILARITY means no Groq, no local Ollama, no
    hallucination risk — just the canned 'no encontré' answer."""
    monkeypatch.setattr(rag, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(rag.ollama, "embed", _fake_embed([1.0, 0.0]))
    monkeypatch.setattr(
        rag.store, "all_chunks", lambda: [_chunk("a.mp4", "no relacionado", [0.0, 1.0])]
    )

    def boom(_prompt):
        raise AssertionError("no LLM should be called below the similarity threshold")

    monkeypatch.setattr(rag, "_answer_groq", boom)
    monkeypatch.setattr(rag, "_answer_local", boom)

    answer, sources, backend = rag.answer_question("pregunta sin relacion")

    assert answer == rag.NO_CONTEXT_ANSWER
    assert sources == []
    assert backend is None


def test_answer_question_excludes_individual_matches_below_the_threshold(monkeypatch):
    """A weak match must not ride along into the prompt/sources just because
    _search's top-K still returned it — even when the BEST match clears
    RAG_MIN_SIMILARITY, every other cited source has to clear it too, or it
    gets cited as 'relevant' despite being unrelated to the question."""
    monkeypatch.setattr(rag, "GROQ_API_KEY", "")
    monkeypatch.setattr(rag, "RAG_TOP_K", 5)
    monkeypatch.setattr(rag.ollama, "embed", _fake_embed([1.0, 0.0]))
    monkeypatch.setattr(
        rag.store, "all_chunks",
        lambda: [
            _chunk(
                "nota-x.txt", "presupuesto de acme, primera cuota vence el 30/09",
                [1.0, 0.0],
            ),
            _chunk("reunion.mkv", "contenido no relacionado", [0.0, 1.0]),  # similarity 0.0
        ],
    )
    seen = {}

    def fake_answer_local(prompt):
        seen["prompt"] = prompt
        return "la cuota vence el 30/09"

    monkeypatch.setattr(rag, "_answer_local", fake_answer_local)

    _answer, sources, _backend = rag.answer_question("cuando vence la cuota de acme?")

    assert sources == [("nota-x.txt", "2026-01-01 00:00:00")]
    assert "reunion.mkv" not in seen["prompt"]
    assert "contenido no relacionado" not in seen["prompt"]


def test_answer_question_uses_groq_when_available(monkeypatch):
    """Groq succeeds: its text is used directly, with sources from the
    matched chunks, and local Ollama never runs."""
    monkeypatch.setattr(rag, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(rag.ollama, "embed", _fake_embed([1.0, 0.0]))
    monkeypatch.setattr(
        rag.store, "all_chunks",
        lambda: [_chunk("reunion.mp4", "presupuesto aprobado", [1.0, 0.0])],
    )
    seen = {}

    def fake_answer_groq(prompt):
        seen["prompt"] = prompt
        return "respuesta de groq"

    monkeypatch.setattr(rag, "_answer_groq", fake_answer_groq)

    def boom(_prompt):
        raise AssertionError("local Ollama must not run when Groq succeeds")

    monkeypatch.setattr(rag, "_answer_local", boom)

    answer, sources, backend = rag.answer_question("que paso con el presupuesto?")

    assert answer == "respuesta de groq"
    assert sources == [("reunion.mp4", "2026-01-01 00:00:00")]
    assert backend == f"Groq · {rag.GROQ_MODEL}"
    assert "presupuesto aprobado" in seen["prompt"]
    assert "que paso con el presupuesto?" in seen["prompt"]


def test_answer_question_falls_back_to_local_on_groq_failure(monkeypatch, read_log):
    """A Groq failure falls back to local Ollama, and the reason is logged."""
    monkeypatch.setattr(rag, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(rag.ollama, "embed", _fake_embed([1.0, 0.0]))
    monkeypatch.setattr(
        rag.store, "all_chunks", lambda: [_chunk("reunion.mp4", "presupuesto", [1.0, 0.0])]
    )

    def raise_error(_prompt):
        raise RuntimeError("groq unavailable")

    monkeypatch.setattr(rag, "_answer_groq", raise_error)
    monkeypatch.setattr(rag, "_answer_local", lambda _prompt: "respuesta local")

    answer, sources, backend = rag.answer_question("pregunta")

    assert answer == "respuesta local"
    assert sources == [("reunion.mp4", "2026-01-01 00:00:00")]
    assert backend == f"local · Ollama {rag.QWEN_MODEL}"
    assert "groq unavailable" in read_log()


def test_answer_question_no_api_key_skips_groq(monkeypatch):
    """Without GROQ_API_KEY (the sandbox default), Groq is never attempted."""
    assert rag.GROQ_API_KEY == ""
    monkeypatch.setattr(rag.ollama, "embed", _fake_embed([1.0, 0.0]))
    monkeypatch.setattr(
        rag.store, "all_chunks", lambda: [_chunk("reunion.mp4", "presupuesto", [1.0, 0.0])]
    )

    def boom(_prompt):
        raise AssertionError("Groq must not run without an API key")

    monkeypatch.setattr(rag, "_answer_groq", boom)
    monkeypatch.setattr(rag, "_answer_local", lambda _prompt: "respuesta local")

    answer, _sources, backend = rag.answer_question("pregunta")

    assert answer == "respuesta local"
    assert backend.startswith("local ·")


def test_answer_question_returns_sorted_unique_source_filenames(monkeypatch):
    """Sources are deduplicated (a file can contribute both a transcript and a
    summary chunk) and sorted by filename; when a file's own chunks carry
    different timestamps, the most recent one wins."""
    monkeypatch.setattr(rag, "GROQ_API_KEY", "")
    monkeypatch.setattr(rag, "RAG_TOP_K", 5)
    monkeypatch.setattr(rag.ollama, "embed", _fake_embed([1.0, 0.0]))
    monkeypatch.setattr(
        rag.store, "all_chunks",
        lambda: [
            _chunk("zeta.mp4", "uno", [1.0, 0.0], created_at="2026-01-01 00:00:00"),
            _chunk(
                "alfa.mp4", "dos", [1.0, 0.0], kind="summary",
                created_at="2026-01-02 00:00:00",
            ),
            _chunk("alfa.mp4", "tres", [1.0, 0.0], created_at="2026-01-01 00:00:00"),
        ],
    )
    monkeypatch.setattr(rag, "_answer_local", lambda _prompt: "ok")

    _answer, sources, _backend = rag.answer_question("pregunta")

    assert sources == [
        ("alfa.mp4", "2026-01-02 00:00:00"),
        ("zeta.mp4", "2026-01-01 00:00:00"),
    ]


def test_sources_dedupes_by_filename_keeping_the_most_recent_timestamp():
    """_sources collapses multiple chunks from the same file into one entry,
    keeping whichever timestamp is most recent, sorted by filename."""
    matches = [
        (_chunk("a.mp4", "x", [1.0, 0.0], created_at="2026-01-01 00:00:00"), 0.9),
        (_chunk("a.mp4", "y", [1.0, 0.0], created_at="2026-01-03 00:00:00"), 0.8),
        (_chunk("b.mp4", "z", [1.0, 0.0], created_at="2026-01-02 00:00:00"), 0.7),
    ]

    assert rag._sources(matches) == [
        ("a.mp4", "2026-01-03 00:00:00"),
        ("b.mp4", "2026-01-02 00:00:00"),
    ]


def test_build_prompt_includes_the_context_the_question_and_each_sources_date():
    """Sanity check on prompt assembly: retrieved text, its source AND
    timestamp, and the original question all make it into the final
    prompt — the model needs the date to reason about recency."""
    matches = [(
        _chunk(
            "reunion.mp4", "se aprobo el presupuesto", [1.0, 0.0],
            created_at="2026-03-15 10:00:00",
        ),
        0.9,
    )]

    prompt = rag._build_prompt("cuanto se aprobo?", matches)

    assert "se aprobo el presupuesto" in prompt
    assert "[Fuente: reunion.mp4 — 2026-03-15 10:00:00]" in prompt
    assert "cuanto se aprobo?" in prompt


def test_answer_groq_uses_the_configured_model(monkeypatch):
    """_answer_groq sends the prompt to Groq and returns its stripped content."""
    monkeypatch.setattr(rag, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(rag, "Groq", lambda **_kw: _FakeGroqClient("  respuesta  "))

    assert rag._answer_groq("prompt") == "respuesta"


def test_answer_local_uses_the_configured_qwen_model(monkeypatch):
    """_answer_local calls ollama.generate with QWEN_MODEL and strips the result."""
    seen = {}

    def fake_generate(model, **_kwargs):
        seen["model"] = model
        return {"response": "  respuesta local  "}

    monkeypatch.setattr(rag.ollama, "generate", fake_generate)

    assert rag._answer_local("prompt") == "respuesta local"
    assert seen["model"] == rag.QWEN_MODEL
