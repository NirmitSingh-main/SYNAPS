/**
 * CopilotMessage — renders a single chat bubble.
 */

interface CopilotMessageProps {
  role: "user" | "assistant";
  content: string;
  isTyping?: boolean;
}

export function CopilotMessage({ role, content, isTyping }: CopilotMessageProps) {
  const isUser = role === "user";

  return (
    <div
      className={`flex gap-2 ${isUser ? "justify-end" : "justify-start"}`}
      style={{ marginBottom: "0.65rem" }}
    >
      {!isUser && (
        <div
          aria-hidden
          style={{
            flexShrink: 0,
            width: 22,
            height: 22,
            borderRadius: "50%",
            background: "var(--color-signal)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginTop: 2,
            fontSize: 10,
            color: "var(--color-background)",
            fontFamily: "var(--font-mono)",
            fontWeight: 700,
            letterSpacing: 0,
          }}
        >
          S
        </div>
      )}

      <div
        style={{
          maxWidth: "82%",
          padding: isUser ? "0.5rem 0.8rem" : "0.55rem 0.85rem",
          borderRadius: isUser ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
          background: isUser
            ? "var(--color-signal)"
            : "color-mix(in srgb, var(--color-surface) 70%, var(--color-background))",
          color: isUser ? "var(--color-background)" : "var(--color-foreground)",
          border: isUser
            ? "none"
            : "1px solid var(--color-border)",
          fontSize: "0.8rem",
          lineHeight: 1.55,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          fontFamily: "var(--font-sans)",
          boxShadow: "0 2px 8px -2px rgba(0,0,0,0.18)",
        }}
      >
        {isTyping ? (
          <span
            aria-label="Copilot is typing"
            style={{
              display: "inline-flex",
              gap: 3,
              alignItems: "center",
              height: 14,
            }}
          >
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                style={{
                  width: 5,
                  height: 5,
                  borderRadius: "50%",
                  background: "var(--color-muted-foreground)",
                  animation: "copilot-dot-bounce 1.2s infinite",
                  animationDelay: `${i * 0.2}s`,
                  display: "inline-block",
                }}
              />
            ))}
          </span>
        ) : (
          content
        )}
      </div>
    </div>
  );
}
