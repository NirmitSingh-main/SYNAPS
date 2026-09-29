/**
 * CopilotButton — global floating action button.
 * Renders bottom-right over every page, opens/closes CopilotPanel.
 */

import { useState } from "react";
import { MessageCircle, X } from "lucide-react";
import { CopilotPanel } from "./CopilotPanel";

export function CopilotButton() {
  const [open, setOpen] = useState(false);

  return (
    <>
      {open && <CopilotPanel onClose={() => setOpen(false)} />}

      <button
        id="copilot-fab"
        type="button"
        aria-label={open ? "Close SYNAPS Copilot" : "Open SYNAPS Copilot"}
        aria-expanded={open}
        aria-controls="copilot-panel"
        onClick={() => setOpen((o) => !o)}
        style={{
          position: "fixed",
          bottom: 24,
          right: 24,
          zIndex: 999,
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "0.5rem 1rem 0.5rem 0.75rem",
          borderRadius: 40,
          border: "1px solid var(--color-border-strong)",
          background: "color-mix(in srgb, var(--color-surface) 70%, var(--color-background))",
          color: "var(--color-foreground)",
          cursor: "pointer",
          backdropFilter: "blur(10px)",
          boxShadow:
            "0 4px 20px -2px rgba(0,0,0,0.28), 0 1px 4px -1px rgba(0,0,0,0.14)",
          transition:
            "transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease",
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLButtonElement).style.transform = "translateY(-2px)";
          (e.currentTarget as HTMLButtonElement).style.boxShadow =
            "0 8px 28px -4px rgba(0,0,0,0.36), 0 2px 6px -1px rgba(0,0,0,0.2)";
          (e.currentTarget as HTMLButtonElement).style.borderColor =
            "var(--color-signal)";
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLButtonElement).style.transform = "translateY(0)";
          (e.currentTarget as HTMLButtonElement).style.boxShadow =
            "0 4px 20px -2px rgba(0,0,0,0.28), 0 1px 4px -1px rgba(0,0,0,0.14)";
          (e.currentTarget as HTMLButtonElement).style.borderColor =
            "var(--color-border-strong)";
        }}
      >
        <span
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: 26,
            height: 26,
            borderRadius: "50%",
            background: "var(--color-signal)",
            flexShrink: 0,
            transition: "transform 200ms ease",
            transform: open ? "rotate(90deg)" : "rotate(0deg)",
          }}
        >
          {open ? (
            <X size={13} color="var(--color-background)" strokeWidth={2.5} />
          ) : (
            <MessageCircle size={13} color="var(--color-background)" strokeWidth={2.5} />
          )}
        </span>

        <span
          style={{
            fontSize: "0.78rem",
            fontFamily: "var(--font-sans)",
            fontWeight: 500,
            letterSpacing: "-0.01em",
            whiteSpace: "nowrap",
          }}
        >
          SYNAPS Copilot
        </span>
      </button>
    </>
  );
}
