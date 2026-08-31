interface Prediction {
  classLabel: string;
  probability: number;
}

interface PredictionChartProps {
  predictions: Prediction[];
  isDemoData?: boolean;
}

export function PredictionChart({ predictions, isDemoData = true }: PredictionChartProps) {
  // Sort descending
  const sorted = [...predictions].sort((a, b) => b.probability - a.probability);

  return (
    <div className="border border-border bg-background">
      <div className="flex items-center justify-between border-b border-border px-6 py-4">
        <p className="label-mono">Classification breakdown</p>
        {isDemoData && (
          <span className="font-mono text-[0.65rem] tracking-[0.14em] text-muted-foreground">
            ILLUSTRATIVE
          </span>
        )}
      </div>

      <ul className="divide-y divide-border" aria-label="Signal classification probabilities">
        {sorted.map((pred, i) => {
          const pct = Math.round(pred.probability * 100);
          const isTop = i === 0;

          return (
            <li key={pred.classLabel} className="px-6 py-4">
              <div className="flex items-center justify-between gap-4">
                <span
                  className="text-[0.85rem] transition-colors"
                  style={{ color: isTop ? "var(--foreground)" : "var(--muted-foreground)" }}
                >
                  {pred.classLabel}
                </span>
                <span
                  className="font-mono text-[0.75rem] shrink-0 tabular-nums"
                  style={{ color: isTop ? "var(--signal)" : "var(--muted-foreground)" }}
                >
                  {pct}%
                </span>
              </div>
              {/* Bar track */}
              <div className="mt-2.5 h-px w-full bg-border">
                <div
                  className="h-px transition-[width] duration-700 ease-out"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: isTop ? "var(--signal)" : "var(--border-strong)",
                  }}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
