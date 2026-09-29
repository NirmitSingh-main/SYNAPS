/**
 * CopilotInput — chat text area with Send button.
 * Enter to send, Shift+Enter for newline.
 */

import { useRef, type KeyboardEvent } from "react";
import { Send } from "lucide-react";

interface CopilotInputProps {
  value: string;
  onChange: (val: string) => void;
  onSend: () => void;
  disabled: boolean;
}

export function CopilotInput({ value, onChange, onSend, disabled }: CopilotInputProps) {
  const taRef = useRef<HTMLTextAreaElement>(null);

  function handleKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) onSend();
    }
  }

  return (
    <div
      style={{
        display: "flex",
        gap: 8,
        alignItems: "flex-end",
        padding: "0.65rem 0.85rem",
        borderTop: "1px solid var(--color-border)",
        background: "color-mix(in srgb, var(--color-surface) 50%, var(--color-background))",
        borderRadius: "0 0 18px 18px",
      }}
    >
      <textarea
        ref={taRef}
        id="copilot-input"
        aria-label="Copilot message input"
        rows={1}
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
          // auto-resize
          const ta = taRef.current;
          if (ta) {
            ta.style.height = "auto";
            ta.style.height = `${Math.min(ta.scrollHeight, 100)}px`;
          }
        }}
        onKeyDown={handleKey}
        placeholder="Ask about this signal, RF theory, or SYNAPS…"
        disabled={disabled}
        style={{
          flex: 1,
          resize: "none",
          overflow: "hidden",
          background: "transparent",
          border: "1px solid var(--color-border)",
          borderRadius: 10,
          padding: "0.45rem 0.7rem",
          fontSize: "0.78rem",
          fontFamily: "var(--font-sans)",
          color: "var(--color-foreground)",
          lineHeight: 1.5,
          outline: "none",
          transition: "border-color 150ms ease",
        }}
        onFocus={(e) => {
          e.target.style.borderColor = "var(--color-border-strong)";
        }}
        onBlur={(e) => {
          e.target.style.borderColor = "var(--color-border)";
        }}
      />

      <button
        id="copilot-send-btn"
        type="button"
        aria-label="Send message"
        onClick={onSend}
        disabled={disabled || !value.trim()}
        style={{
          flexShrink: 0,
          width: 34,
          height: 34,
          borderRadius: "50%",
          background:
            disabled || !value.trim()
              ? "var(--color-secondary)"
              : "var(--color-signal)",
          border: "none",
          cursor: disabled || !value.trim() ? "not-allowed" : "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          transition: "background 150ms ease, transform 120ms ease",
          transform: "translateY(0)",
        }}
        onMouseEnter={(e) => {
          if (!disabled && value.trim())
            (e.currentTarget as HTMLButtonElement).style.transform =
              "translateY(-1px)";
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLButtonElement).style.transform = "translateY(0)";
        }}
      >
        <Send
          size={14}
          color={
            disabled || !value.trim()
              ? "var(--color-muted-foreground)"
              : "var(--color-background)"
          }
        />
      </button>
    </div>
  );
}
