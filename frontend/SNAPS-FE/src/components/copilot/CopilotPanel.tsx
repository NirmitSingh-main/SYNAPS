/**
 * CopilotPanel — floating chat panel (bottom-right).
 * Context-aware: reads current page, analysis state, and auth session.
 */

import {
  useEffect,
  useRef,
  useState,
  useCallback,
  type KeyboardEvent,
} from "react";
import { X } from "lucide-react";
import { useLocation } from "@tanstack/react-router";
import { useAuth } from "@/lib/AuthContext";
import { getAnalysisData } from "@/lib/analysisStore";
import {
  sendCopilotMessage,
  buildAnalysisContext,
  type CopilotMessage as ICopilotMessage,
} from "@/services/copilot";
import { CopilotMessage } from "./CopilotMessage";
import { CopilotInput } from "./CopilotInput";

// ── Quick prompts per page ────────────────────────────────────────────────────

const QUICK_PROMPTS: Record<string, string[]> = {
  "/": [
    "What is SYNAPS?",
    "How does SYNAPS work?",
    "What is an IQ signal?",
  ],
  "/analyze": [
    "How do I analyze a signal?",
    "What does SNR mean?",
    "Explain the analysis pipeline",
  ],
  "/results": [
    "Explain this result",
    "Why was this modulation detected?",
    "What does confidence mean?",
  ],
  "/processing": [
    "What is SYNAPS analyzing right now?",
    "How long does analysis take?",
  ],
  "/dashboard": [
    "Summarize my analyses",
    "What modulation do I see most?",
    "What was my latest analysis?",
  ],
  "/history": [
    "What modulations have I analyzed?",
    "Where can I see my past results?",
    "How do I filter my history?",
  ],
  "/admin": [
    "What is the admin dashboard?",
    "How do I navigate SYNAPS?",
  ],
};

function pageLabel(pathname: string): string {
  const map: Record<string, string> = {
    "/": "home",
    "/analyze": "analyze",
    "/results": "results",
    "/processing": "processing",
    "/dashboard": "dashboard",
    "/history": "history",
    "/admin": "admin",
    "/auth": "auth",
  };
  return map[pathname] ?? "unknown";
}

// ── Panel ─────────────────────────────────────────────────────────────────────

interface CopilotPanelProps {
  onClose: () => void;
}

const WELCOME_MESSAGE: ICopilotMessage = {
  role: "assistant",
  content:
    "Hello! I'm SYNAPS Copilot — your RF Signal Intelligence assistant.\n\nAsk me about your current analysis, signal theory, modulation, or how to use SYNAPS.",
};

export function CopilotPanel({ onClose }: CopilotPanelProps) {
  const location = useLocation();
  const { user, session } = useAuth();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [messages, setMessages] = useState<ICopilotMessage[]>([WELCOME_MESSAGE]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const currentPage = pageLabel(location.pathname);
  const quickPrompts = QUICK_PROMPTS[location.pathname] ?? QUICK_PROMPTS["/"] ?? [];

  // Auto-scroll to newest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Dismiss on Escape
  useEffect(() => {
    function onKey(e: globalThis.KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const send = useCallback(
    async (overrideMessage?: string) => {
      const msg = (overrideMessage ?? input).trim();
      if (!msg || loading) return;

      if (!overrideMessage) setInput("");

      const userMsg: ICopilotMessage = { role: "user", content: msg };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);

      // Build context — strip binary arrays from analysis
      const rawAnalysis = getAnalysisData() as Record<string, unknown> | null;
      const analysisCtx =
        currentPage === "results" || currentPage === "processing"
          ? buildAnalysisContext(rawAnalysis)
          : null;

      // Conversation history (exclude welcome message)
      const historyForApi: ICopilotMessage[] = messages
        .slice(1) // skip static welcome
        .concat(userMsg)
        .slice(-12); // keep last 12 turns

      try {
        const resp = await sendCopilotMessage(
          {
            message: msg,
            page: currentPage,
            analysis_context: analysisCtx as Record<string, unknown> | null,
            history: historyForApi.slice(0, -1), // don't include current msg again
          },
          session?.access_token
        );

        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: resp.response },
        ]);
      } catch (err) {
        const errMsg =
          err instanceof Error ? err.message : "Unexpected error.";
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Copilot is temporarily unavailable. Please try again in a moment.\n\n_${errMsg}_`,
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [input, loading, messages, currentPage, session]
  );

  const showQuickPrompts =
    messages.length === 1 && !loading; // only show on fresh open

  return (
    <div
      id="copilot-panel"
      role="dialog"
      aria-label="SYNAPS Copilot chat panel"
      aria-modal="true"
      style={{
        position: "fixed",
        bottom: 88,
        right: 24,
        width: "min(380px, calc(100vw - 32px))",
        height: "min(520px, calc(100vh - 120px))",
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        borderRadius: 18,
        border: "1px solid var(--color-border-strong)",
        background: "color-mix(in srgb, var(--color-background) 92%, var(--color-surface))",
        boxShadow:
          "0 8px 40px -4px rgba(0,0,0,0.32), 0 2px 8px -2px rgba(0,0,0,0.18)",
        backdropFilter: "blur(12px)",
        animation: "copilot-panel-in 220ms cubic-bezier(0.2, 0.8, 0.3, 1) both",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0.75rem 1rem",
          borderBottom: "1px solid var(--color-border)",
          flexShrink: 0,
        }}
      >
        <div>
          <p
            style={{
              margin: 0,
              fontSize: "0.82rem",
              fontWeight: 600,
              fontFamily: "var(--font-sans)",
              color: "var(--color-foreground)",
              letterSpacing: "-0.01em",
            }}
          >
            SYNAPS Copilot
          </p>
          <p
            style={{
              margin: 0,
              marginTop: 1,
              fontSize: "0.65rem",
              fontFamily: "var(--font-mono)",
              color: "var(--color-muted-foreground)",
              letterSpacing: "0.06em",
            }}
          >
            RF Signal Intelligence Assistant
            {user ? "" : " · Sign in for history"}
          </p>
        </div>
        <button
          id="copilot-panel-close"
          type="button"
          aria-label="Close Copilot"
          onClick={onClose}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            color: "var(--color-muted-foreground)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 4,
            borderRadius: 6,
            transition: "color 150ms ease",
          }}
          onMouseEnter={(e) =>
            (e.currentTarget.style.color = "var(--color-foreground)")
          }
          onMouseLeave={(e) =>
            (e.currentTarget.style.color = "var(--color-muted-foreground)")
          }
        >
          <X size={16} />
        </button>
      </div>

      {/* Messages */}
      <div
        id="copilot-messages"
        aria-live="polite"
        aria-label="Chat messages"
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "0.85rem 0.9rem",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {messages.map((m, i) => (
          <CopilotMessage key={i} role={m.role} content={m.content} />
        ))}

        {/* Typing indicator */}
        {loading && <CopilotMessage role="assistant" content="" isTyping />}

        {/* Quick prompts — shown only on open */}
        {showQuickPrompts && (
          <div
            style={{
              marginTop: "0.6rem",
              display: "flex",
              flexWrap: "wrap",
              gap: 6,
            }}
          >
            {quickPrompts.map((prompt) => (
              <button
                key={prompt}
                type="button"
                onClick={() => send(prompt)}
                style={{
                  background: "color-mix(in srgb, var(--color-surface) 60%, var(--color-background))",
                  border: "1px solid var(--color-border)",
                  borderRadius: 20,
                  padding: "0.3rem 0.7rem",
                  fontSize: "0.72rem",
                  fontFamily: "var(--font-sans)",
                  color: "var(--color-muted-foreground)",
                  cursor: "pointer",
                  transition: "border-color 150ms ease, color 150ms ease, transform 120ms ease",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.borderColor =
                    "var(--color-border-strong)";
                  (e.currentTarget as HTMLButtonElement).style.color =
                    "var(--color-foreground)";
                  (e.currentTarget as HTMLButtonElement).style.transform =
                    "translateY(-1px)";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.borderColor =
                    "var(--color-border)";
                  (e.currentTarget as HTMLButtonElement).style.color =
                    "var(--color-muted-foreground)";
                  (e.currentTarget as HTMLButtonElement).style.transform =
                    "translateY(0)";
                }}
              >
                {prompt}
              </button>
            ))}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <CopilotInput
        value={input}
        onChange={setInput}
        onSend={() => send()}
        disabled={loading}
      />
    </div>
  );
}
