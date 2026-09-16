"""Summary generation.

Tries Groq's hosted chat model first (skipped entirely when no
``GROQ_API_KEY`` is configured) and falls back to the local Qwen model served
by Ollama on any failure — network, timeout, rate limit, context-length, or
anything else. Every attempt and fallback is logged.
"""

import ollama
from groq import Groq, RateLimitError

from config import (
    GROQ_API_KEY,
    GROQ_CHUNK_CHARS,
    GROQ_CHUNK_MAX_COMPLETION_TOKENS,
    GROQ_MAX_COMPLETION_TOKENS,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
    QWEN_MODEL,
)
from utils import log, log_warning

# The prompt's wording/line lengths are content, not code — kept verbatim.
# pylint: disable-next=line-too-long
_PROMPT_TEMPLATE = """IMPORTANTE: tu respuesta completa debe estar SIEMPRE en español, sin excepción, sin importar en qué idioma esté la transcripción. No respondas en inglés ni en ningún otro idioma bajo ninguna circunstancia.

Analiza la transcripción completa que aparece al final de este mensaje y crea un resumen detallado, fiel y estructurado. No respondas sobre el proceso de análisis: entrega directamente el resumen final.

## Objetivo
Conservar todos los detalles relevantes de la conversación sin convertir el resultado en una transcripción literal. Incluye los puntos clave tratados, decisiones, argumentos importantes, compromisos, problemas críticos y problemas resueltos. Cuando sea posible, indica qué persona mencionó cada punto.

## Instrucciones de análisis
1. Lee y analiza toda la transcripción antes de redactar el resumen.
2. No inventes información, nombres, responsabilidades, fechas, decisiones ni conclusiones.
3. Diferencia claramente entre hechos afirmados, propuestas, dudas, opiniones, decisiones confirmadas y asuntos pendientes.
4. Atribuye cada punto a la persona correspondiente únicamente cuando el nombre o la identidad del hablante estén disponibles o puedan inferirse con seguridad.
5. Si un punto fue mencionado por varias personas, indícalo cuando sea relevante.
6. Conserva fechas, cifras, nombres de proyectos, productos, clientes, sistemas, plazos, dependencias y cualquier otro dato concreto presente en la transcripción.
7. Los números de ticket son números enteros de 4 dígitos (por ejemplo 3619). Escríbelos siempre completos y juntos, sin puntos, espacios ni barras — nunca "36.19", "36/19" ni "36 19".
8. Mantén el contexto necesario para que cada punto pueda entenderse sin consultar la transcripción original.
9. Señala contradicciones, ambigüedades o información insuficiente sin intentar resolverlas por cuenta propia.
10. No confundas una intención o sugerencia con un compromiso confirmado.
11. Si una sección no está respaldada por la transcripción, omítela por completo. No escribas frases como «no se mencionó», «sin información» o similares.
12. Si la transcripción contiene errores, interrupciones o frases incompletas, interpreta solo lo que pueda determinarse razonablemente y marca como incierto cualquier aspecto dudoso.
13. Evita repetir el mismo contenido en varias secciones, salvo que sea necesario para relacionar un problema con su resolución o con un compromiso.

## Formato de salida
Usa Markdown claro, profesional y en español (nunca en inglés ni en otro idioma). Incluye únicamente las secciones que tengan contenido respaldado por la transcripción, siguiendo este orden preferente:

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
- Todo número de ticket aparece completo y junto (4 dígitos, sin puntos ni barras).

## Transcripción completa

--- INICIO DE LA TRANSCRIPCIÓN ---
{text}
--- FIN DE LA TRANSCRIPCIÓN ---
"""

# Groq's free tier enforces a strict output-tokens-per-minute cap (observed
# ~1000 for qwen/qwen3.8-27b) that the detailed prompt above routinely exceeds
# for a real meeting. This shorter prompt is used ONLY for the Groq attempt, to
# fit a complete answer under that cap instead of triggering the truncation
# fallback below on almost every dense meeting. Local Ollama has no such
# per-minute limit, so it always gets the detailed prompt above instead.
# pylint: disable-next=line-too-long
_PROMPT_TEMPLATE_GROQ = """Analiza la transcripción que aparece al final y devuelve un resumen breve y fiel en español, en Markdown. No inventes nombres, fechas ni datos que no estén en el texto; atribuí una afirmación a alguien solo si su identidad es clara por nombre o contexto. Si algo no se puede determinar con seguridad, omitilo — no lo adivines.

Sé conciso: máximo 3-4 viñetas por sección, sin relleno ni repetición.

Los números de ticket son números enteros de 4 dígitos (ej. 3619). Escríbelos
siempre completos y juntos, sin puntos, espacios ni barras — nunca "36.19",
"36/19" ni "36 19".

Incluí SIEMPRE los cinco encabezados de abajo, exactamente en ese orden y con
ese texto, aunque una sección no tenga contenido en la transcripción — en ese
caso escribí una sola línea "Sin información en la transcripción." debajo del
encabezado. Nunca omitas un encabezado.

## Resumen
1-2 párrafos cortos: propósito de la reunión y resultado general.

## Puntos clave
- Lo más importante tratado, con el hablante si se sabe.

## Decisiones
- Solo decisiones confirmadas (no propuestas ni dudas).

## Compromisos
- Acción — responsable — plazo (si consta).

## Pendientes
- Temas abiertos o próximos pasos.

Transcripción:
--- INICIO ---
{text}
--- FIN ---
"""

# generate_summary treats a response missing any of these as incomplete (see
# _groq_chat) — the prompt above requires the model to always emit all five,
# even as a "sin información" placeholder, so a missing header reliably means
# the model didn't follow instructions or got cut off, not that the section
# legitimately had nothing to say.
_GROQ_REQUIRED_HEADERS = (
    "## Resumen", "## Puntos clave", "## Decisiones", "## Compromisos", "## Pendientes",
)

# Used only for a transcript longer than GROQ_CHUNK_CHARS (see
# _summarize_groq_chunked): one of these runs per chunk, asking for terse,
# uncategorized-format notes rather than the full 5-section structure — that
# structure only gets imposed once, in the synthesis prompt below, on the
# already-condensed notes.
# pylint: disable-next=line-too-long
_PROMPT_TEMPLATE_GROQ_CHUNK = """Este es el fragmento {index}/{total} de la transcripción de UNA reunión — no es la reunión completa, así que no la trates como si lo fuera y no le pongas encabezados Markdown (#, ##) a tu respuesta.

En español, en viñetas breves, extraé SOLO lo que aparece explícitamente en este fragmento, agrupado en estas categorías (omití una categoría por completo si no tiene nada en este fragmento):

PUNTOS: lo más importante tratado, con el hablante si se sabe.
DECISIONES: solo decisiones confirmadas (no propuestas ni dudas).
COMPROMISOS: acción — responsable — plazo (si consta).
RIESGOS: problemas o riesgos mencionados.
PENDIENTES: temas abiertos o próximos pasos.

No inventes nada, no repitas la transcripción, sé conciso.

Fragmento:
--- INICIO ---
{text}
--- FIN ---
"""

# Combines the per-chunk notes above into one final structured summary — same
# required-headers contract as _PROMPT_TEMPLATE_GROQ, applied once to the
# already-condensed notes instead of the raw transcript.
# pylint: disable-next=line-too-long
_PROMPT_TEMPLATE_GROQ_SYNTHESIS = """A continuación hay notas extraídas fragmento por fragmento de la transcripción completa de UNA reunión larga, en orden. Combinalas en un solo resumen final en español, fusionando en un solo punto lo que se repita entre fragmentos en vez de listarlo varias veces. No inventes nada que no esté en las notas.

Los números de ticket son números enteros de 4 dígitos (ej. 3619). Escríbelos
siempre completos y juntos, sin puntos, espacios ni barras — nunca "36.19",
"36/19" ni "36 19".

Incluí SIEMPRE los cinco encabezados de abajo, exactamente en ese orden y con
ese texto, aunque una sección no tenga contenido en las notas — en ese caso
escribí una sola línea "Sin información en la transcripción." debajo del
encabezado. Nunca omitas un encabezado.

## Resumen
1-2 párrafos cortos: propósito de la reunión y resultado general.

## Puntos clave
- Lo más importante tratado, con el hablante si se sabe.

## Decisiones
- Solo decisiones confirmadas (no propuestas ni dudas).

## Compromisos
- Acción — responsable — plazo (si consta).

## Pendientes
- Temas abiertos o próximos pasos.

Notas parciales:
--- INICIO ---
{text}
--- FIN ---
"""


def _split_into_chunks(text, max_chars):
    """Split ``text`` into chunks no longer than ``max_chars``.

    Breaks on the nearest earlier newline so a chunk never cuts a line in
    half; falls back to a hard cut only when a single line exceeds
    ``max_chars`` on its own (mirrors ``telegram_api.split_message``, kept
    separate since the two have no real reason to share code).
    """
    chunks = []
    while len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


def _summarize_local(prompt):
    """Summarize with the local Qwen model via Ollama."""
    response = ollama.generate(model=QWEN_MODEL, prompt=prompt)
    # pylint mis-infers ollama.generate's return type; the mapping access is fine.
    return response["response"].strip()  # pylint: disable=no-member


# qwen/qwen3.8-27b is a reasoning model: its (discarded) chain-of-thought and
# its visible answer draw from the SAME max_completion_tokens budget. If the
# reasoning eats most of it, the model can emit a short, incomplete-looking
# answer and still stop "cleanly" (finish_reason="stop") right at the edge of
# the cap, instead of Groq reporting "length". Treating completion_tokens
# landing this close to the cap as a truncation too — not just an explicit
# "length" — catches that case as well.
_GROQ_NEAR_CAP_RATIO = 0.95


def _groq_chat(client, prompt, max_completion_tokens, *, require_headers, label):
    """Run one Groq chat completion and apply the shared truncation checks.

    Uses ``reasoning_format="parsed"`` so ``message.content`` holds only the
    final answer; any reasoning trace the model produces goes to
    ``message.reasoning`` and is discarded.

    A response that looks incomplete is treated as a failure rather than
    returned as-is, so the caller falls back instead of silently delivering a
    truncated or partial summary. Three signals trigger that: an explicit
    ``finish_reason != "stop"`` (e.g. ``"length"``); ``completion_tokens``
    landing within ``_GROQ_NEAR_CAP_RATIO`` of ``max_completion_tokens`` even
    though Groq reported a clean stop; or, when ``require_headers`` is set —
    observed in practice with plenty of unused token budget and a clean
    ``"stop"`` — the model simply skipping one of the section headers
    ``_GROQ_REQUIRED_HEADERS`` requires, which no token-based check can catch
    since nothing was actually cut short.

    Args:
        client: An already-constructed ``groq.Groq`` client.
        prompt: The full prompt text.
        max_completion_tokens: Output token budget for this specific request
            (smaller for a chunk than for a single-shot summary or the final
            synthesis — see :func:`_summarize_groq_chunked`).
        require_headers: Whether to also require ``_GROQ_REQUIRED_HEADERS``.
            Off for a chunk request, which intentionally uses a different,
            uncategorized note format (see ``_PROMPT_TEMPLATE_GROQ_CHUNK``).
        label: Short tag for this call's log line and any error it raises
            (e.g. ``"summary"``, ``"chunk 2/4"``, ``"synthesis"``).

    Raises:
        Exception: Any failure from the Groq client (network, timeout, rate
            limit, context length, invalid key, truncated or incomplete
            response, ...). Callers are expected to fall back to
            :func:`_summarize_local`.
    """
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=max_completion_tokens,
        reasoning_format="parsed",
    )
    choice = completion.choices[0]
    completion_tokens = getattr(getattr(completion, "usage", None), "completion_tokens", None)
    log(
        f"Groq {label}: finish_reason={choice.finish_reason!r} "
        f"completion_tokens={completion_tokens}/{max_completion_tokens}"
    )
    near_cap = (
        completion_tokens is not None
        and completion_tokens >= max_completion_tokens * _GROQ_NEAR_CAP_RATIO
    )
    if choice.finish_reason != "stop" or near_cap:
        raise RuntimeError(
            f"{label} response truncated (finish_reason={choice.finish_reason!r}, "
            f"completion_tokens={completion_tokens}/{max_completion_tokens})"
        )
    content = choice.message.content.strip()
    if require_headers:
        missing = [h for h in _GROQ_REQUIRED_HEADERS if h not in content]
        if missing:
            raise RuntimeError(
                f"{label} response truncated: missing section(s) {', '.join(missing)}"
            )
    return content


def _summarize_groq(prompt):
    """Summarize with Groq's hosted chat model in a single request.

    Used for a transcript no longer than ``GROQ_CHUNK_CHARS``; a longer one
    goes through :func:`_summarize_groq_chunked` instead. See
    :func:`_groq_chat` for the truncation checks applied to the response.
    """
    client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS)
    return _groq_chat(
        client, prompt, GROQ_MAX_COMPLETION_TOKENS, require_headers=True, label="summary"
    )


def _summarize_groq_chunked(text):
    """Summarize a transcript too long for one Groq request.

    Splits ``text`` into chunks of at most ``GROQ_CHUNK_CHARS``, asks Groq for
    terse, uncategorized notes on each (a small output budget per chunk —
    ``GROQ_CHUNK_MAX_COMPLETION_TOKENS``), then makes one more Groq call to
    synthesize those notes into the final structured summary.

    Why this helps at all: Groq's free-tier OTPM (output-tokens-per-minute)
    cap rejects a request outright if its estimated output looks too large,
    and that estimate scales with the transcript — a single request for a
    long, dense meeting can be rejected even with the already-concise prompt.
    Each chunk needs far less output than one request for the whole
    transcript would, so each is far less likely to be rejected on its own.

    What this does NOT solve: OTPM is a per-MINUTE cap, not a per-request one.
    Enough chunks fired close together can still add up to more than one
    request's worth of output within the same window — this raises the
    ceiling on how long a meeting Groq can handle, it does not remove the
    limit. A transcript long enough to need many chunks can still exhaust the
    per-minute budget partway through and fall back to local, same as before.

    Raises:
        Exception: If any chunk or the synthesis call fails. The whole
            attempt is aborted rather than risk mixing partial Groq content
            with a local fallback — the caller re-summarizes the FULL
            transcript locally instead (see :func:`generate_summary`).
    """
    chunks = _split_into_chunks(text, GROQ_CHUNK_CHARS)
    log(f"Transcript is {len(text)} chars; splitting into {len(chunks)} chunks for Groq.")
    client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS)

    partials = []
    for index, chunk in enumerate(chunks, start=1):
        prompt = _PROMPT_TEMPLATE_GROQ_CHUNK.format(index=index, total=len(chunks), text=chunk)
        partials.append(
            _groq_chat(
                client, prompt, GROQ_CHUNK_MAX_COMPLETION_TOKENS,
                require_headers=False, label=f"chunk {index}/{len(chunks)}",
            )
        )

    notes = "\n\n".join(
        f"[Fragmento {i}/{len(chunks)}]\n{partial}" for i, partial in enumerate(partials, start=1)
    )
    synthesis_prompt = _PROMPT_TEMPLATE_GROQ_SYNTHESIS.format(text=notes)
    return _groq_chat(
        client, synthesis_prompt, GROQ_MAX_COMPLETION_TOKENS,
        require_headers=True, label="synthesis",
    )


def generate_summary(text):
    """Summarize a transcript into a Spanish summary.

    Groq gets a short, concise prompt sized to fit its free-tier output-tokens-
    per-minute (OTPM) limit, sent as a single request first. If Groq rejects
    that request specifically for being too large for OTPM (``RateLimitError``
    — a pre-flight rejection based on the request's *estimated* output, which
    scales with transcript content and not just its length; a transcript well
    under ``GROQ_CHUNK_CHARS`` can still trigger it), it's retried chunked
    (see :func:`_summarize_groq_chunked`) instead of falling back immediately.
    Any other Groq failure — or a chunked retry that also fails — falls back
    to local Ollama (no OTPM limit), which gets the full detailed prompt
    (executive synthesis, key points, decisions, commitments with
    owner/deadline, risks, resolved issues, open items, and discrepancies).
    Both attribute statements to a speaker only when the transcript makes
    that identifiable, and never invent names, dates, or facts not present in
    the text.

    Args:
        text: The transcript (or note) to summarize; may be in any language.

    Returns:
        A ``(summary, backend_label)`` tuple, e.g. ``(summary, "Groq · qwen/qwen3.8-27b")``
        or ``(summary, "local · Ollama qwen3:1.7b")``.
    """
    if GROQ_API_KEY:
        try:
            log(f"Generating summary with Groq ({GROQ_MODEL})...")
            groq_prompt = _PROMPT_TEMPLATE_GROQ.format(text=text)
            summary = _summarize_groq(groq_prompt)
            log("Groq summary succeeded.")
            return summary, f"Groq · {GROQ_MODEL}"
        except RateLimitError as e:
            log_warning(f"Groq rejected the request as too large for OTPM ({e}); retrying chunked.")
            try:
                summary = _summarize_groq_chunked(text)
                log("Groq summary succeeded (chunked).")
                return summary, f"Groq · {GROQ_MODEL}"
            except Exception as e2:  # noqa: BLE001  pylint: disable=broad-exception-caught
                log_warning(
                    f"Chunked Groq summary also failed ({e2}); falling back to local Ollama."
                )
        except Exception as e:  # noqa: BLE001  pylint: disable=broad-exception-caught
            log_warning(f"Groq summary failed ({e}); falling back to local Ollama.")

    log(f"Generating summary with local Ollama ({QWEN_MODEL})...")
    local_prompt = _PROMPT_TEMPLATE.format(text=text)
    return _summarize_local(local_prompt), f"local · Ollama {QWEN_MODEL}"
