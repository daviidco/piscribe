"""Tests for embeddings.py: sentence-bounded chunking and RAG indexing."""

# Reaches into the module's own private helpers on purpose, same as test_summary.py.
# pylint: disable=protected-access

from datetime import datetime, timezone

import embeddings
import store


def test_chunk_text_short_text_is_one_chunk():
    """Text under the limit comes back as a single chunk, unchanged."""
    assert embeddings._chunk_text("hola", max_chars=100) == ["hola"]


def test_chunk_text_splits_at_sentence_boundaries_and_preserves_content():
    """A long, sentence-rich text splits into within-limit chunks that each
    end on a full sentence, losing nothing."""
    sentences = [f"Esta es la oracion numero {i}." for i in range(30)]
    text = " ".join(sentences)
    chunks = embeddings._chunk_text(text, max_chars=80, overlap_chars=0)

    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)
    assert all(c.endswith(".") for c in chunks)
    assert " ".join(chunks) == text


def test_chunk_text_hard_cuts_when_no_sentence_boundary_fits():
    """With no sentence end (or newline) to break on, the splitter falls
    back to a hard character cut, same as summary._split_into_chunks."""
    text = "x" * 9000
    chunks = embeddings._chunk_text(text, max_chars=4000, overlap_chars=0)

    assert [len(c) for c in chunks] == [4000, 4000, 1000]


def test_chunk_text_repeats_overlap_chars_between_consecutive_chunks():
    """Each chunk after the first repeats the tail of the previous one, so a
    point made right at a cut lands in both chunks instead of being lost."""
    text = "a" * 15 + ". " + "b" * 15 + ". " + "c" * 15
    chunks = embeddings._chunk_text(text, max_chars=20, overlap_chars=5)

    assert chunks == [
        "a" * 15 + ".",
        "aaa. " + "b" * 15 + ".",
        "bbb. " + "c" * 15,
    ]


class _FakeEmbedResponse:  # pylint: disable=too-few-public-methods
    """Stand-in for ollama.EmbedResponse — just the .embeddings attribute."""

    def __init__(self, vectors):
        self.embeddings = vectors


def test_embed_texts_calls_ollama_embed_once_for_the_whole_batch(monkeypatch):
    """A batch of texts is embedded in a single Ollama call, not one per text."""
    calls = []

    def fake_embed(model, input):  # pylint: disable=redefined-builtin
        calls.append((model, input))
        return _FakeEmbedResponse([[0.1, 0.2] for _ in input])

    monkeypatch.setattr(embeddings.ollama, "embed", fake_embed)

    vectors = embeddings._embed_texts(["uno", "dos", "tres"])

    assert len(calls) == 1
    assert calls[0] == (embeddings.EMBED_MODEL, ["uno", "dos", "tres"])
    assert vectors == [[0.1, 0.2]] * 3


def test_index_file_indexes_both_transcript_and_summary(monkeypatch):
    """index_file(transcript=..., summary=...) (re)indexes both kinds."""
    monkeypatch.setattr(embeddings, "_embed_texts", lambda texts: [[1.0] for _ in texts])
    calls = []
    monkeypatch.setattr(
        store, "replace_chunks",
        lambda filename, kind, chunks: calls.append((filename, kind, chunks)),
    )

    embeddings.index_file("reunion.mp4", transcript="texto de transcripcion", summary="resumen")

    assert {kind for _f, kind, _c in calls} == {"transcript", "summary"}
    assert all(f == "reunion.mp4" for f, _k, _c in calls)


def test_index_file_with_only_summary_does_not_touch_transcript_chunks(monkeypatch):
    """index_file(summary=...) alone (used by _resummarize) never calls
    replace_chunks for 'transcript' — those chunks are left as-is."""
    monkeypatch.setattr(embeddings, "_embed_texts", lambda texts: [[1.0] for _ in texts])
    calls = []
    monkeypatch.setattr(
        store, "replace_chunks",
        lambda filename, kind, chunks: calls.append((filename, kind, chunks)),
    )

    embeddings.index_file("reunion.mp4", summary="resumen nuevo")

    assert [kind for _f, kind, _c in calls] == ["summary"]


def test_index_file_with_neither_argument_does_nothing(monkeypatch):
    """Nothing to index means no Ollama call and no store write at all."""

    def boom(**_kw):
        raise AssertionError("must not call ollama.embed with nothing to index")

    monkeypatch.setattr(embeddings.ollama, "embed", boom)
    calls = []
    monkeypatch.setattr(store, "replace_chunks", lambda *a, **kw: calls.append(a))

    embeddings.index_file("reunion.mp4")

    assert not calls


def test_index_file_pairs_each_chunk_with_its_own_embedding_in_order(monkeypatch):
    """replace_chunks gets (text, vector) pairs matching by position, not just
    the raw chunk/vector lists side by side."""
    monkeypatch.setattr(embeddings, "_chunk_text", lambda text, **_kw: [f"{text}-1", f"{text}-2"])
    monkeypatch.setattr(
        embeddings, "_embed_texts", lambda texts: [[float(i)] for i in range(len(texts))]
    )
    seen = {}
    monkeypatch.setattr(
        store, "replace_chunks", lambda filename, kind, chunks: seen.setdefault(kind, chunks)
    )

    embeddings.index_file("reunion.mp4", transcript="T", summary="S")

    assert seen["transcript"] == [("T-1", [0.0]), ("T-2", [1.0])]
    assert seen["summary"] == [("S-1", [0.0]), ("S-2", [1.0])]


def test_index_note_labels_and_stores_with_kind_manual(monkeypatch):
    """index_note (used by /input) chunks + embeds the free text and stores
    it under kind='manual', labeled with a UTC timestamp, and returns that
    label so the caller can tell the user where the note landed."""
    monkeypatch.setattr(embeddings, "_embed_texts", lambda texts: [[1.0] for _ in texts])
    seen = {}
    monkeypatch.setattr(
        store, "replace_chunks",
        lambda filename, kind, chunks: seen.update(filename=filename, kind=kind, chunks=chunks),
    )

    label = embeddings.index_note("nota pegada de otra reunion.")

    assert label.startswith("nota-")
    assert seen["filename"] == label
    assert seen["kind"] == "manual"
    assert seen["chunks"] == [("nota pegada de otra reunion.", [1.0])]


def test_index_note_never_overwrites_a_previous_note(monkeypatch):
    """Two separate /input calls at different timestamps get two distinct
    labels — each note is its own additive RAG source, never replacing an
    earlier one via the same (filename, kind) replace_chunks would collide on."""
    monkeypatch.setattr(embeddings, "_embed_texts", lambda texts: [[1.0] for _ in texts])
    calls = []
    monkeypatch.setattr(
        store, "replace_chunks", lambda filename, kind, chunks: calls.append(filename)
    )

    class _FakeDatetime:  # pylint: disable=too-few-public-methods
        """Stand-in for datetime.datetime with a scripted sequence of .now() values."""

        _values = [
            datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        ]

        @classmethod
        def now(cls, _tz):
            """Pop the next scripted timestamp."""
            return cls._values.pop(0)

    monkeypatch.setattr(embeddings, "datetime", _FakeDatetime)

    first = embeddings.index_note("primera nota")
    second = embeddings.index_note("segunda nota")

    assert first != second
    assert calls == [first, second]
