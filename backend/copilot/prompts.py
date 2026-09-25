"""
SYNAPS Copilot — Prompt Builder
Constructs Groq-ready system and user message context from intents,
knowledge base, and runtime data (current analysis, user history).
"""

from __future__ import annotations
from typing import Any

from backend.copilot.knowledge import (
    SYNAPS_KNOWLEDGE,
    RF_SIGNAL_THEORY,
    NAVIGATION_GUIDE,
    PAGE_CONTEXTS,
)
from backend.copilot.intents import (
    GENERAL_GREETING,
    RF_THEORY,
    SYNAPS_KNOWLEDGE as INTENT_SYNAPS,
    NAVIGATION,
    CURRENT_ANALYSIS,
    USER_HISTORY,
    GENERAL,
)

# ── System persona ─────────────────────────────────────────────────────────────
PERSONA = """You are SYNAPS Copilot, the intelligent assistant for the SYNAPS RF Signal Intelligence Platform.

Your role:
- Answer questions about RF signals, signal theory, modulation, DSP, and the SYNAPS platform
- Help users understand their specific analysis results using the actual data provided
- Guide users through navigation and features
- Be concise, accurate, and technically precise
- Respond naturally to greetings and casual conversation

Rules:
- NEVER invent or fabricate analysis values, measurements, or statistics
- ONLY use values from the analysis data provided to you
- If a piece of information is not in the provided context, say so honestly
- Do not expose internal system details, API keys, database credentials, or other users' data
- Keep responses focused and appropriately concise — technical but readable
- Use plain English for explanations; you may use markdown for structure when helpful
"""

# ── Context section builders ───────────────────────────────────────────────────

def _format_analysis_context(analysis: dict[str, Any]) -> str:
    """Format the current analysis result into a structured context block."""
    if not analysis:
        return ""

    lines = ["=== CURRENT ANALYSIS RESULT ==="]
    lines.append(f"Filename: {analysis.get('filename', 'unknown')}")
    lines.append(f"Format: {analysis.get('format', 'IQ')}")
    lines.append(f"Classification: {analysis.get('classification', '—')}")
    
    conf = analysis.get("confidence")
    if conf is not None:
        lines.append(f"Confidence: {round(conf * 100, 1)}%")

    lines.append(f"Sample Rate: {_fmt_hz(analysis.get('sampleRate', 0))}")
    lines.append(f"Duration: {_fmt_ms(analysis.get('duration', 0))}")
    lines.append(f"Bandwidth (99%): {_fmt_hz(analysis.get('bandwidth', 0))}")
    
    snr = analysis.get("snr")
    if snr is not None:
        lines.append(f"SNR: {snr:.1f} dB")
    
    lines.append(f"Peak Frequency: {_fmt_hz(analysis.get('peakFrequency', 0))}")
    lines.append(f"Samples: {analysis.get('numSamples', 0):,}")

    # Prediction breakdown
    preds = analysis.get("predictionBreakdown") or []
    if preds:
        lines.append("\nPrediction Probabilities:")
        for p in preds:
            lines.append(f"  {p.get('classLabel', '?')}: {round(p.get('probability', 0) * 100, 1)}%")

    # DSP features
    features = analysis.get("features") or []
    if features:
        lines.append("\nExtracted DSP Features:")
        for f in features:
            lines.append(f"  {f.get('name', '?')}: {f.get('value', '?')} {f.get('unit', '')}".rstrip())

    # Bit recovery
    br = analysis.get("bitRecovery")
    if br:
        lines.append(f"\nBit Recovery:")
        lines.append(f"  Validation Status: {br.get('validation_status', '—')}")
        lines.append(f"  Reference Bits: {br.get('reference_bit_count', '—')}")
        lines.append(f"  Recovered Bits: {br.get('recovered_bit_count', '—')}")
        acc = br.get("bit_accuracy_pct")
        if acc is not None:
            lines.append(f"  Bit Accuracy: {acc:.1f}%")
        ber = br.get("ber")
        if ber is not None:
            lines.append(f"  BER: {ber:.4f}")

    # Explanation
    explanation = analysis.get("explanation")
    if explanation:
        lines.append(f"\nAnalysis Explanation:\n{explanation}")

    return "\n".join(lines)


def _format_history_context(history: list[dict[str, Any]] | None) -> str:
    """Format user analysis history into a structured context block."""
    if history is None:
        return (
            "=== USER ANALYSIS HISTORY ===\n"
            "The user is not signed in / unauthenticated. If the user asks about their personal history, "
            "saved analyses, or statistics, politely inform them that they must sign in to view their analysis history."
        )

    if len(history) == 0:
        return (
            "=== USER ANALYSIS HISTORY ===\n"
            "Authenticated user has 0 saved analysis records in their history. "
            "They have not analyzed or saved any signals yet."
        )

    lines = [f"=== USER ANALYSIS HISTORY ({len(history)} records in total) ==="]

    from collections import Counter
    mods = Counter(r.get("classification", "Unknown") for r in history)
    most_common_mod, most_common_count = mods.most_common(1)[0] if mods else ("None", 0)

    # Calculate overall averages
    confidences = [
        float(r["confidence"])
        for r in history
        if r.get("confidence") is not None
    ]
    avg_conf = (sum(confidences) / len(confidences) * 100) if confidences else None

    snrs = [
        float(r["snr"])
        for r in history
        if r.get("snr") is not None
    ]
    avg_snr = (sum(snrs) / len(snrs)) if snrs else None

    lines.append(f"Total analyses: {len(history)}")
    lines.append(f"Most frequent modulation: {most_common_mod} ({most_common_count} times)")
    lines.append(f"Modulation breakdown: {dict(mods)}")

    if avg_conf is not None:
        lines.append(f"Average confidence: {avg_conf:.1f}%")
    if avg_snr is not None:
        lines.append(f"Average SNR: {avg_snr:.2f} dB")

    # Latest record
    latest = history[0]
    latest_conf_str = f"{round((float(latest.get('confidence') or 0)) * 100, 1)}%" if latest.get('confidence') is not None else "N/A"
    latest_snr_str = f"{float(latest.get('snr')):.1f} dB" if latest.get('snr') is not None else "N/A"
    lines.append(
        f"Latest analysis: \"{latest.get('filename', 'unknown')}\" | "
        f"Modulation: {latest.get('classification', '—')} | "
        f"Confidence: {latest_conf_str} | "
        f"SNR: {latest_snr_str} | "
        f"Date: {str(latest.get('created_at', ''))[:19]}"
    )

    # Extrema: SNR
    with_snr = [r for r in history if r.get("snr") is not None]
    if with_snr:
        best_snr_rec = max(with_snr, key=lambda r: float(r["snr"]))
        worst_snr_rec = min(with_snr, key=lambda r: float(r["snr"]))
        lines.append(
            f"Highest SNR: {float(best_snr_rec['snr']):.2f} dB (File: \"{best_snr_rec.get('filename', '?')}\", Mod: {best_snr_rec.get('classification', '?')})"
        )
        lines.append(
            f"Lowest SNR: {float(worst_snr_rec['snr']):.2f} dB (File: \"{worst_snr_rec.get('filename', '?')}\", Mod: {worst_snr_rec.get('classification', '?')})"
        )

    # Extrema: Confidence
    with_conf = [r for r in history if r.get("confidence") is not None]
    if with_conf:
        best_conf_rec = max(with_conf, key=lambda r: float(r["confidence"]))
        lines.append(
            f"Highest confidence: {float(best_conf_rec['confidence']) * 100:.1f}% (File: \"{best_conf_rec.get('filename', '?')}\", Mod: {best_conf_rec.get('classification', '?')})"
        )

    # Per-modulation breakdown
    lines.append("\nPer-Modulation Statistics:")
    for mod_name in sorted(mods.keys()):
        mod_records = [r for r in history if r.get("classification") == mod_name]
        m_confs = [float(r["confidence"]) for r in mod_records if r.get("confidence") is not None]
        m_snrs = [float(r["snr"]) for r in mod_records if r.get("snr") is not None]
        m_avg_c = f"{sum(m_confs)/len(m_confs)*100:.1f}%" if m_confs else "N/A"
        m_avg_s = f"{sum(m_snrs)/len(m_snrs):.1f} dB" if m_snrs else "N/A"
        lines.append(f"  • {mod_name}: {len(mod_records)} analyses | Avg Conf: {m_avg_c} | Avg SNR: {m_avg_s}")

    # List up to all 30 records
    lines.append(f"\nAll Analysis Records (newest to oldest):")
    for idx, r in enumerate(history[:30], 1):
        c_val = f"{round((float(r.get('confidence') or 0)) * 100, 1)}%" if r.get('confidence') is not None else "N/A"
        s_val = f"{float(r.get('snr')):.1f} dB" if r.get('snr') is not None else "N/A"
        sr_val = _fmt_hz(r.get("sample_rate") or r.get("sampleRate"))
        bw_val = _fmt_hz(r.get("bandwidth"))
        dt_val = str(r.get("created_at", ""))[:16].replace("T", " ")
        lines.append(
            f"  {idx}. [{dt_val}] {r.get('filename', '?')} | {r.get('classification', '?')} | Conf: {c_val} | SNR: {s_val} | SR: {sr_val} | BW: {bw_val}"
        )

    return "\n".join(lines)


def build_system_prompt(
    intents: list[str],
    page: str = "unknown",
    analysis: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """
    Build the full system prompt from intents, page context, analysis data, and history.
    """
    sections: list[str] = [PERSONA]

    # Page context
    page_ctx = PAGE_CONTEXTS.get(page, PAGE_CONTEXTS["unknown"])
    sections.append(f"=== CURRENT PAGE CONTEXT ===\n{page_ctx}")

    # Knowledge sections based on intents
    if GENERAL_GREETING in intents or GENERAL in intents:
        # Keep it light — just SYNAPS brief
        sections.append(f"=== SYNAPS OVERVIEW ===\n{SYNAPS_KNOWLEDGE[:1200]}")

    if INTENT_SYNAPS in intents:
        sections.append(f"=== SYNAPS PLATFORM KNOWLEDGE ===\n{SYNAPS_KNOWLEDGE}")

    if RF_THEORY in intents:
        sections.append(f"=== RF SIGNAL THEORY ===\n{RF_SIGNAL_THEORY}")

    if NAVIGATION in intents:
        sections.append(f"=== NAVIGATION GUIDE ===\n{NAVIGATION_GUIDE}")

    if CURRENT_ANALYSIS in intents and analysis:
        sections.append(_format_analysis_context(analysis))

    if USER_HISTORY in intents:
        sections.append(_format_history_context(history))

    return "\n\n".join(sections)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_hz(hz: float | None) -> str:
    if not hz:
        return "0 Hz"
    if abs(hz) >= 1_000_000:
        return f"{hz / 1_000_000:.2f} MHz"
    if abs(hz) >= 1_000:
        return f"{hz / 1_000:.1f} kHz"
    return f"{hz:.0f} Hz"


def _fmt_ms(s: float | None) -> str:
    if not s:
        return "0 ms"
    return f"{s * 1000:.1f} ms"


# ── Public builder used by service.py ──────────────────────────────────────────

def build_copilot_messages(
    user_message: str,
    page: str | None = None,
    analysis: dict[str, Any] | None = None,
    user_history: list[dict[str, Any]] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """
    Assemble a Groq-compatible messages list:
      [system, ...conversation_history, user]

    The system prompt is constructed from page context, intents, and runtime data.
    """
    from backend.copilot.intents import classify_intent

    # Determine which knowledge sections are relevant
    intents = classify_intent(user_message, page)

    system_prompt = build_system_prompt(
        intents=intents,
        page=page or "unknown",
        analysis=analysis,
        history=user_history,
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Append short rolling conversation history (last 10 turns max)
    if conversation_history:
        for turn in conversation_history[-10:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages
