"""Answering /ask questions from indexed transcripts/summaries.

Retrieval (embedding the question, comparing it against indexed chunks)
always runs through local Ollama, same as embeddings.py — Groq has no
embeddings API. Drafting the answer itself tries Groq's hosted chat model
first (skipped entirely when no GROQ_API_KEY is configured) and falls back
to the local Qwen model on any failure, same policy as
summary.generate_summary.
"""

import ollama
from groq import Groq

import store
from config import (
    EMBED_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
    QWEN_MODEL,
    RAG_MIN_SIMILARITY,
    RAG_TOP_K,
)
from utils import log, log_warning

NO_CONTEXT_ANSWER = "No encontré información sobre eso en las reuniones indexadas."

_ANSWER_TEMPERATURE = 0.2

# pylint: disable-next=line-too-long
_PROMPT_TEMPLATE = """Respondé la pregunta del usuario en español, basándote ÚNICAMENTE en los fragmentos de reuniones que aparecen abajo. No inventes ni agregues información que no esté en esos fragmentos — si no alcanzan para responder con seguridad, decilo explícitamente en vez de adivinar.

Al final de tu respuesta, citá de qué archivo salió cada dato relevante, usando el nombre exacto que aparece en cada fragmento, con el formato "Fuente: archivo".

Fragmentos:
--- INICIO ---
{context}
--- FIN ---

Pregunta: {question}
"""


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _search(question_embedding):
    """Up to ``RAG_TOP_K`` ``(chunk, similarity)`` pairs, best match first.

    Brute-force over every indexed chunk — no vector index (e.g. sqlite-vec):
    at this scale (a handful of meetings) a full scan is already fast.
    """
    scored = [
        (chunk, _cosine_similarity(question_embedding, chunk["embedding"]))
        for chunk in store.all_chunks()
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:RAG_TOP_K]


def _build_prompt(question, matches):
    context = "\n\n".join(
        f"[Fuente: {chunk['filename']}]\n{chunk['text']}" for chunk, _ in matches
    )
    return _PROMPT_TEMPLATE.format(context=context, question=question)


def _answer_local(prompt):
    """Answer with the local Qwen model via Ollama."""
    response = ollama.generate(model=QWEN_MODEL, prompt=prompt)
    # pylint mis-infers ollama.generate's return type; the mapping access is fine.
    return response["response"].strip()  # pylint: disable=no-member


def _answer_groq(prompt):
    """Answer with Groq's hosted chat model in a single request."""
    client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=_ANSWER_TEMPERATURE,
    )
    return completion.choices[0].message.content.strip()


def answer_question(question):
    """Answer ``question`` from the indexed transcript/summary chunks.

    Returns a ``(answer, sources, backend)`` tuple. ``sources`` is a sorted
    list of the unique filenames the answer draws from — empty when nothing
    was relevant enough. ``backend`` is ``None`` when the best match didn't
    cross ``RAG_MIN_SIMILARITY``: no LLM is called at all in that case,
    avoiding both the cost and the risk of an invented answer.
    """
    response = ollama.embed(model=EMBED_MODEL, input=question)
    # pylint mis-infers ollama.embed's return type, same as ollama.generate in
    # summary.py; the .embeddings access is fine.
    question_embedding = response.embeddings[0]  # pylint: disable=no-member
    matches = _search(list(question_embedding))
    if not matches or matches[0][1] < RAG_MIN_SIMILARITY:
        return NO_CONTEXT_ANSWER, [], None

    prompt = _build_prompt(question, matches)
    sources = sorted({chunk["filename"] for chunk, _ in matches})

    if GROQ_API_KEY:
        try:
            log(f"Answering /ask with Groq ({GROQ_MODEL})...")
            answer = _answer_groq(prompt)
            log("Groq /ask answer succeeded.")
            return answer, sources, f"Groq · {GROQ_MODEL}"
        except Exception as e:  # noqa: BLE001  pylint: disable=broad-exception-caught
            log_warning(f"/ask Groq answer failed ({e}); falling back to local Ollama.")

    log(f"Answering /ask with local Ollama ({QWEN_MODEL})...")
    return _answer_local(prompt), sources, f"local · Ollama {QWEN_MODEL}"
