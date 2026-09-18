"""Chunking and local embeddings for the /ask RAG index.

Groq has no embeddings API, so unlike summary.py/video.py this module has no
Groq branch at all — indexing always runs through local Ollama
(config.EMBED_MODEL), regardless of whether GROQ_API_KEY is configured.
"""

import ollama

import store
from config import EMBED_MODEL, RAG_CHUNK_CHARS, RAG_CHUNK_OVERLAP_CHARS


def _chunk_text(text, max_chars=RAG_CHUNK_CHARS, overlap_chars=RAG_CHUNK_OVERLAP_CHARS):
    """Split ``text`` into sentence-bounded chunks of at most ``max_chars``
    (plus overlap; see below).

    Unlike ``summary._split_into_chunks`` (paragraph/line boundaries, tuned
    for Groq's per-request output budget), retrieval benefits from cutting at
    sentence boundaries: a chunk should read as one or more complete
    thoughts, not an arbitrary paragraph fragment. The cut point prefers the
    latest sentence end (``". "``, ``"! "``, ``"? "``) within range, falls
    back to any newline, and only hard-cuts mid-sentence when a stretch that
    long has neither.

    Each chunk after the first also repeats the last ``overlap_chars``
    characters of the previous one, same rationale as
    ``summary._split_into_chunks``: a point made right at a cut shouldn't
    land entirely on one side.
    """
    chunks = []
    tail = ""
    while len(text) > max_chars:
        window = text[:max_chars]
        cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
        if cut > 0:
            cut += 2
        else:
            cut = window.rfind("\n")
        if cut <= 0:
            cut = max_chars
        raw = text[:cut]
        chunk = (tail + raw).strip()
        if chunk:
            chunks.append(chunk)
        tail = raw[-overlap_chars:] if overlap_chars else ""
        text = text[cut:].lstrip()
    chunk = (tail + text).strip()
    if chunk:
        chunks.append(chunk)
    return chunks


def _embed_texts(texts):
    """Embed a batch of texts in a single Ollama call, one vector per text."""
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    # pylint mis-infers ollama.embed's return type, same as ollama.generate in
    # summary.py; the .embeddings access is fine.
    return list(response.embeddings)  # pylint: disable=no-member


def index_file(filename, transcript=None, summary=None):
    """(Re)index ``filename``'s transcript and/or summary chunks for /ask.

    Either argument may be omitted (``None``/empty) to leave that kind's
    existing chunks untouched — e.g. ``_resummarize`` passes only ``summary``
    since the transcript didn't change.
    """
    for text, kind in ((transcript, "transcript"), (summary, "summary")):
        if not text:
            continue
        chunks = _chunk_text(text)
        vectors = _embed_texts(chunks)
        store.replace_chunks(filename, kind, list(zip(chunks, vectors)))
