"""
SYNAPS Copilot — Intent Classifier
Classifies incoming user messages into semantic intents to select
the correct knowledge/context for Groq prompt construction.
"""

from __future__ import annotations
import re

# Intent constants
GENERAL_GREETING = "GENERAL_GREETING"
RF_THEORY = "RF_THEORY"
SYNAPS_KNOWLEDGE = "SYNAPS_KNOWLEDGE"
NAVIGATION = "NAVIGATION"
CURRENT_ANALYSIS = "CURRENT_ANALYSIS"
USER_HISTORY = "USER_HISTORY"
GENERAL = "GENERAL"

# ── Keyword pattern sets ───────────────────────────────────────────────────────
_GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|howdy|greetings|what'?s up|good (morning|afternoon|evening)|how are you|how's it going|sup)\b",
    re.IGNORECASE,
)

_RF_KEYWORDS = re.compile(
    r"\b(iq|in-phase|quadrature|bpsk|qpsk|fsk|qam|qam16|modulation|demodulation|"
    r"cumulant|c40|c42|papr|snr|signal.to.noise|bandwidth|sample.?rate|nyquist|"
    r"fft|spectrum|spectrogram|psd|frequency|carrier|cfo|offset|phase|amplitude|"
    r"symbol.?rate|bit.?rate|ber|bit.?error|constellation|waveform|rf.?finger|"
    r"fingerprint|transformer|classifier|confidence|probability|synchronization|"
    r"synchronize|decod|recovered.?bit|bit.?recovery|occupied.?bandwidth|noise.?floor|"
    r"antenna|radio|sdr|software.?defined|sampling|aliasing|harmonics|sidebands)\b",
    re.IGNORECASE,
)

_SYNAPS_KEYWORDS = re.compile(
    r"\b(synaps|platform|pipeline|backend|frontend|upload|analyze|analysis|"
    r"how does synaps|what is synaps|this system|this app|this platform|"
    r"how does it work|what can you do|what does synaps|dsp|signal intelligence|"
    r"classification|auto.*modulation|recognition)\b",
    re.IGNORECASE,
)

_NAVIGATION_KEYWORDS = re.compile(
    r"\b(how do i|navigate|go to|where (can|do) i|find|access|open|"
    r"sign in|sign up|login|log in|log out|sign out|register|account|"
    r"dashboard|history|results|analyze page|upload page|admin|"
    r"my previous|my past|my account|where is|which page|nav)\b",
    re.IGNORECASE,
)

_CURRENT_ANALYSIS_KEYWORDS = re.compile(
    r"\b(this result|this analysis|my result|this signal|this file|"
    r"current result|explain (my|this|the) (result|analysis|signal|classification)|"
    r"why (was|is|did) (this|my)|what does (this|my) (result|confidence|snr|classification) mean|"
    r"explain (the )?(snr|confidence|bandwidth|waveform|spectrum|spectrogram|"
    r"classification|prediction|feature|cfo|papr|cumulant|fingerprint|bit recovery|ber))\b",
    re.IGNORECASE,
)

_USER_HISTORY_KEYWORDS = re.compile(
    r"\b("
    r"my\s+(analyses|analysis|history|past|previous|recent|results|signals|records|uploads|runs|tests)|"
    r"(past|previous|recent|all)\s+(analyses|analysis|results|signals|records|uploads|runs)|"
    r"how\s+many\s+(analyses|signals|records|files|uploads|times|qam|bpsk|qpsk|fsk|runs)|"
    r"how\s+many\s+.*(analyz|process|classif|upload)\w*|"
    r"(latest|last|recent|newest)\s+(analysis|result|signal|classification|run)|"
    r"(what|which)\s+(was|is)\s+my\s+(latest|last|recent|first)|"
    r"(what\s+did\s+i|signals\s+i|analyses\s+i|have\s+i)\s+(analyz|upload|run|process)\w*|"
    r"(highest|lowest|best|worst|top|max|min)\s+(snr|confidence|ber|accuracy|bandwidth)|"
    r"average\s+(confidence|snr|ber|accuracy|duration|bandwidth)|"
    r"(most\s+common|most\s+frequent|predominant|top)\s+modulation|"
    r"modulation\s+.*(appear|seen|occur)\w*\s+(most|often|frequently)|"
    r"(what|which)\s+modulation.*(appear|seen|occur|most)|"
    r"(summarize|summary\s+of|overview\s+of|list|show|compare)\s+(all\s+)?my|"
    r"compare\s+my\s+.*(analyses|signals|qpsk|qam|bpsk|fsk)|"
    r"(qam16|qpsk|bpsk|fsk|qam)\s+signals?\s+i\s+(have\s+)?analyz\w*|"
    r"total\s+analyses|history\s+summary|my\s+dashboard"
    r")\b",
    re.IGNORECASE,
)


def classify_intent(message: str, page: str = "unknown", has_analysis: bool = False) -> list[str]:
    """
    Classify a user message into one or more intent labels.
    Returns a prioritized list; the first element is the primary intent.
    """
    msg = message.strip()

    intents: list[str] = []

    # Greeting check (exact short messages)
    if _GREETING_PATTERNS.match(msg) and len(msg.split()) <= 8:
        intents.append(GENERAL_GREETING)

    # User history — check before general RF theory so queries like "What is my average SNR?" route to history
    if _USER_HISTORY_KEYWORDS.search(msg) or (
        page in ("history", "dashboard") and re.search(r"\b(summarize|summary|overview|breakdown|stats|statistics|modulations)\b", msg, re.I)
    ):
        intents.append(USER_HISTORY)

    # Current analysis — elevated when on results page or has_analysis context
    if _CURRENT_ANALYSIS_KEYWORDS.search(msg) or (
        page in ("results", "processing") and has_analysis and _RF_KEYWORDS.search(msg)
    ):
        intents.append(CURRENT_ANALYSIS)

    # Navigation
    if _NAVIGATION_KEYWORDS.search(msg):
        intents.append(NAVIGATION)

    # SYNAPS platform knowledge
    if _SYNAPS_KEYWORDS.search(msg):
        intents.append(SYNAPS_KNOWLEDGE)

    # RF / signal theory
    if _RF_KEYWORDS.search(msg):
        intents.append(RF_THEORY)

    # Fall-through
    if not intents:
        intents.append(GENERAL)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for i in intents:
        if i not in seen:
            seen.add(i)
            unique.append(i)

    return unique
