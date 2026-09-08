"""Summary generation with a local Qwen model served by Ollama."""

import ollama

from config import QWEN_MODEL


def generate_summary(text):
    """Summarize a transcript into a structured, concise Spanish summary.

    The prompt is deliberately written in Spanish and instructs the model to
    always answer in Spanish regardless of the source language, to return a
    fixed set of sections (summary, key points, decisions, open items,
    participants), and to attribute statements to a speaker only when the
    transcript makes that identifiable — never inventing names or facts.

    Args:
        text: The transcript (or note) to summarize; may be in any language.

    Returns:
        The model's summary as a stripped string, always in Spanish.
    """
    prompt = f"""Eres un asistente que resume transcripciones de reuniones de trabajo.

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
    response = ollama.generate(model=QWEN_MODEL, prompt=prompt)
    # pylint mis-infers ollama.generate's return type; the mapping access is fine.
    return response["response"].strip()  # pylint: disable=no-member
