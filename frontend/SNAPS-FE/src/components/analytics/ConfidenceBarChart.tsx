import { useState } from "react";

export interface ConfidenceBarItem {
  label: string;
  avgConfidence: number | null;
  sampleCount: number;
}

interface ConfidenceBarChartProps {
  data: ConfidenceBarItem[];
  title?: string;
  subtitle?: string;
}

export function ConfidenceBarChart({
  data,
  title = "Average Confidence",
  subtitle = "Average AI confidence by modulation",
}: ConfidenceBarChartProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const hasAnyData = data.some((d) => d.avgConfidence !== null && d.sampleCount > 0);

  return (
    <div className="floating-surface p-5 h-full flex flex-col justify-between">
      <div>
        <p className="label-mono">{title}</p>
        <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>
      </div>

      {!hasAnyData ? (
        <div className="my-8 flex flex-col items-center justify-center py-6 text-center">
          <p className="font-mono text-xs text-muted-foreground">
            No confidence metrics available
          </p>
        </div>
      ) : (
        <div className="my-4 space-y-3.5">
          {data.map((item, idx) => {
            const isHovered = hoveredIdx === idx;
            const hasVal = item.avgConfidence !== null && item.sampleCount > 0;
            const val = hasVal ? item.avgConfidence! : 0;

            return (
              <div
                key={item.label}
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
                className={`relative rounded-lg p-1.5 transition-colors ${
                  isHovered ? "bg-surface/70" : ""
                }`}
              >
                <div className="flex items-center justify-between font-mono text-xs">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-foreground">{item.label}</span>
                    <span className="text-[0.68rem] text-muted-foreground">
                      ({item.sampleCount} {item.sampleCount === 1 ? "sig" : "sigs"})
                    </span>
                  </div>
                  <span
                    className={`font-mono text-xs ${
                      hasVal ? "text-signal font-semibold" : "text-muted-foreground"
                    }`}
                  >
                    {hasVal ? `${val.toFixed(1)}%` : "—"}
                  </span>
                </div>

                {/* Bar */}
                <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-border/60">
                  {hasVal && (
                    <div
                      className="h-full rounded-full bg-signal transition-all duration-700 ease-out"
                      style={{
                        width: `${Math.min(100, Math.max(0, val))}%`,
                        opacity: isHovered ? 1 : 0.85,
                      }}
                    />
                  )}
                </div>

                {/* Tooltip on hover */}
                {isHovered && hasVal && (
                  <div className="absolute right-2 -top-7 z-10 rounded border border-border-strong bg-background/95 px-2 py-1 font-mono text-[0.65rem] text-foreground shadow-sm">
                    {item.label}: {val.toFixed(2)}% avg ({item.sampleCount} runs)
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
