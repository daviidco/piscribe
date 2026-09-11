"""Summary generation.

Tries Groq's hosted chat model first (skipped entirely when no
``GROQ_API_KEY`` is configured) and falls back to the local Qwen model served
by Ollama on any failure — network, timeout, rate limit, context-length, or
anything else. Every attempt and fallback is logged.
"""

import ollama
from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_TIMEOUT_SECONDS, QWEN_MODEL
from utils import log

_PROMPT_TEMPLATE = """Eres un asistente que resume transcripciones de reuniones de trabajo.

Instrucciones:
- Responde SIEMPRE en español, aunque la transcripción esté en otro idioma.
- Sé claro y conciso. No agregues preámbulos, disculpas ni comentarios sobre el proceso.
- Atribuye afirmaciones, propuestas y compromisos a quién los dijo, pero SOLO cuando
  la transcripción lo permita identificar (por un nombre citado o por el contexto).
  Si no se puede determinar, no lo atribuyas; nunca inventes ni adivines nombres.
- No inventes decisiones, fechas, números de ticket ni ningún dato que no esté en el texto.
- Si el texto no es una reunión o es demasiado breve para resumir, dilo en una frase.

Devuelve exactamente estas secciones, en este orden (omite una sección si no aplica):

Resumen: frases con lo esencial.

Puntos clave:
- <punto> — (quién lo planteó, si consta)

Decisiones:
- <decisión> — (quién la impulsó o aprobó, si consta)

Pendientes:
- <acción> — responsable: <nombre o "sin asignar"> — fecha: <si consta>

Participantes: <nombres que aparezcan en la transcripción, o "no identificados">

Transcripción:
{text}

Resumen:"""


def _summarize_local(prompt):
    """Summarize with the local Qwen model via Ollama."""
    response = ollama.generate(model=QWEN_MODEL, prompt=prompt)
    # pylint mis-infers ollama.generate's return type; the mapping access is fine.
    return response["response"].strip()  # pylint: disable=no-member


def _summarize_groq(prompt):
    """Summarize with Groq's hosted chat model.

    Uses ``reasoning_format="parsed"`` so ``message.content`` holds only the
    final answer; any reasoning trace the model produces goes to
    ``message.reasoning`` and is discarded.

    Raises:
        Exception: Any failure from the Groq client (network, timeout, rate
            limit, context length, invalid key, ...). Callers are expected to
            fall back to :func:`_summarize_local`.
    """
    client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=2048,
        reasoning_format="parsed",
    )
    return completion.choices[0].message.content.strip()


def generate_summary(text):
    """Summarize a transcript into a structured, concise Spanish summary.

    The same prompt is used for both backends: always answer in Spanish
    regardless of the source language, return a fixed set of sections
    (summary, key points, decisions, open items, participants), and attribute
    statements to a speaker only when the transcript makes that identifiable —
    never inventing names or facts.

    Args:
        text: The transcript (or note) to summarize; may be in any language.

    Returns:
        A ``(summary, backend_label)`` tuple, e.g. ``(summary, "Groq · qwen/qwen3.8-27b")``
        or ``(summary, "local · Ollama qwen3:1.7b")``.
    """
    prompt = _PROMPT_TEMPLATE.format(text=text)

    if GROQ_API_KEY:
        try:
            log(f"Generating summary with Groq ({GROQ_MODEL})...")
            summary = _summarize_groq(prompt)
            log("Groq summary succeeded.")
            return summary, f"Groq · {GROQ_MODEL}"
        except Exception as e:  # noqa: BLE001  pylint: disable=broad-exception-caught
            log(f"Groq summary failed ({e}); falling back to local Ollama.")

    log(f"Generating summary with local Ollama ({QWEN_MODEL})...")
    return _summarize_local(prompt), f"local · Ollama {QWEN_MODEL}"
