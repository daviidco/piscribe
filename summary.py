"""Summary generation.

Tries Groq's hosted chat model first (skipped entirely when no
``GROQ_API_KEY`` is configured) and falls back to the local Qwen model served
by Ollama on any failure — network, timeout, rate limit, context-length, or
anything else. Every attempt and fallback is logged.
"""

import ollama
from groq import Groq

from config import (
    GROQ_API_KEY,
    GROQ_MAX_COMPLETION_TOKENS,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
    QWEN_MODEL,
)
from utils import log

# The prompt's wording/line lengths are content, not code — kept verbatim.
# pylint: disable-next=line-too-long
_PROMPT_TEMPLATE = """Analiza la transcripción completa que aparece al final de este mensaje y crea un resumen detallado, fiel y estructurado. No respondas sobre el proceso de análisis: entrega directamente el resumen final.

## Objetivo
Conservar todos los detalles relevantes de la conversación sin convertir el resultado en una transcripción literal. Incluye los puntos clave tratados, decisiones, argumentos importantes, compromisos, problemas críticos y problemas resueltos. Cuando sea posible, indica qué persona mencionó cada punto.

## Instrucciones de análisis
1. Lee y analiza toda la transcripción antes de redactar el resumen.
2. No inventes información, nombres, responsabilidades, fechas, decisiones ni conclusiones.
3. Diferencia claramente entre hechos afirmados, propuestas, dudas, opiniones, decisiones confirmadas y asuntos pendientes.
4. Atribuye cada punto a la persona correspondiente únicamente cuando el nombre o la identidad del hablante estén disponibles o puedan inferirse con seguridad.
5. Si un punto fue mencionado por varias personas, indícalo cuando sea relevante.
6. Conserva fechas, cifras, nombres de proyectos, productos, clientes, sistemas, plazos, dependencias y cualquier otro dato concreto presente en la transcripción.
7. Mantén el contexto necesario para que cada punto pueda entenderse sin consultar la transcripción original.
8. Señala contradicciones, ambigüedades o información insuficiente sin intentar resolverlas por cuenta propia.
9. No confundas una intención o sugerencia con un compromiso confirmado.
10. Si una sección no está respaldada por la transcripción, omítela por completo. No escribas frases como «no se mencionó», «sin información» o similares.
11. Si la transcripción contiene errores, interrupciones o frases incompletas, interpreta solo lo que pueda determinarse razonablemente y marca como incierto cualquier aspecto dudoso.
12. Evita repetir el mismo contenido en varias secciones, salvo que sea necesario para relacionar un problema con su resolución o con un compromiso.

## Formato de salida
Usa Markdown claro, profesional y español. Incluye únicamente las secciones que tengan contenido respaldado por la transcripción, siguiendo este orden preferente:

# Resumen de la reunión o conversación

## Síntesis ejecutiva
Resume en pocos párrafos el propósito, los temas principales y el resultado general.

## Puntos clave tratados
Presenta los temas importantes en viñetas. Para cada punto, indica el hablante cuando sea posible y conserva los detalles relevantes.

## Decisiones y conclusiones
Incluye solo decisiones o conclusiones confirmadas. Indica quién las tomó o mencionó cuando sea posible.

## Compromisos y acciones acordadas
Para cada compromiso, especifica:
- Acción
- Responsable
- Fecha límite o plazo
- Dependencias o condiciones
- Estado, si se conoce

No completes ningún dato ausente con suposiciones.

## Problemas críticos o riesgos
Describe el problema, su impacto, las causas mencionadas, la persona que lo planteó —si se conoce— y el estado actual.

## Problemas resueltos
Indica qué problema se resolvió, cómo se resolvió, quién participó y cualquier condición o seguimiento pendiente.

## Temas pendientes y próximos pasos
Incluye asuntos abiertos, preguntas sin respuesta, decisiones pendientes y acciones futuras.

## Discrepancias, dudas o información ambigua
Incluye esta sección solo si la transcripción contiene contradicciones, afirmaciones dudosas o información que no permite llegar a una conclusión clara.

## Detalles adicionales relevantes
Incluye aquí información importante que no encaje adecuadamente en las secciones anteriores.

## Reglas para la atribución de hablantes
- Si aparecen nombres explícitos, utilízalos tal como figuran.
- Si solo aparecen etiquetas como «Hablante 1» o «Participante A», conserva esas etiquetas.
- Si una persona puede identificarse por contexto pero no con certeza, indica la atribución como «posiblemente [identidad]» o evita atribuirla.
- Nunca atribuyas una afirmación a una persona basándote únicamente en una suposición.

## Verificación final antes de responder
Comprueba que:
- Se ha considerado toda la transcripción.
- No se han omitido detalles relevantes, fechas, cifras o condiciones.
- Cada compromiso está diferenciado de una simple propuesta.
- Los problemas críticos están separados de los problemas resueltos.
- Las atribuciones de hablantes son prudentes y trazables.
- No se ha añadido información que no esté en la transcripción.
- Se han omitido las secciones sin contenido.
- El resumen es detallado, pero no repite innecesariamente la transcripción.

## Transcripción completa

--- INICIO DE LA TRANSCRIPCIÓN ---
{text}
--- FIN DE LA TRANSCRIPCIÓN ---
"""


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

    A response cut short by ``GROQ_MAX_COMPLETION_TOKENS`` (``finish_reason ==
    "length"``) is treated as a failure rather than returned incomplete, so a
    dense, multi-participant meeting falls back to local Ollama instead of
    silently delivering a truncated summary.

    Raises:
        Exception: Any failure from the Groq client (network, timeout, rate
            limit, context length, invalid key, truncated response, ...).
            Callers are expected to fall back to :func:`_summarize_local`.
    """
    client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=GROQ_MAX_COMPLETION_TOKENS,
        reasoning_format="parsed",
    )
    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError(
            f"response truncated at GROQ_MAX_COMPLETION_TOKENS={GROQ_MAX_COMPLETION_TOKENS}"
        )
    return choice.message.content.strip()


def generate_summary(text):
    """Summarize a transcript into a detailed, structured Spanish summary.

    The same prompt is used for both backends: analyze the full transcript and
    produce a Markdown summary (executive synthesis, key points, decisions,
    commitments with owner/deadline, risks, resolved issues, open items, and
    discrepancies), attributing statements to a speaker only when the
    transcript makes that identifiable, and never inventing names, dates, or
    facts not present in the text.

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
