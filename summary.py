"""Summary generation with a local Qwen model served by Ollama."""

import ollama

from config import QWEN_MODEL


def generate_summary(text):
    """Summarize a transcript into a concise Spanish summary.

    The prompt is deliberately written in Spanish and instructs the model to
    always answer in Spanish regardless of the source language, highlighting key
    points, decisions made, and open items.

    Args:
        text: The transcript (or note) to summarize; may be in any language.

    Returns:
        The model's summary as a stripped string, always in Spanish.
    """
    prompt = f"""Eres un asistente que resume transcripciones de reuniones de trabajo.
Resume el siguiente texto en español, de forma clara y concisa, destacando
los puntos clave, decisiones tomadas y pendientes. El texto puede estar en
cualquier idioma; tu resumen siempre debe estar en español.

Texto:
{text}

Resumen en español:"""
    response = ollama.generate(model=QWEN_MODEL, prompt=prompt)
    return response["response"].strip()
