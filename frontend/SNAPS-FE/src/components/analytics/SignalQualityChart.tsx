import { useState } from "react";

export interface SignalQualityItem {
  label: string;
  avgSnr: number | null;
  sampleCount: number;
}

interface SignalQualityChartProps {
  data: SignalQualityItem[];
  title?: string;
  subtitle?: string;
}

export function SignalQualityChart({
  data,
  title = "Signal Quality",
  subtitle = "Average SNR by detected modulation",
}: SignalQualityChartProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const validItems = data.filter((d) => d.avgSnr !== null && d.sampleCount > 0);
  const maxSnr = validItems.length > 0 ? Math.max(...validItems.map((d) => d.avgSnr!), 20) : 30;

  return (
    <div className="floating-surface p-5 h-full flex flex-col justify-between">
      <div>
        <p className="label-mono">{title}</p>
        <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>
      </div>

      {validItems.length === 0 ? (
        <div className="my-8 flex flex-col items-center justify-center py-6 text-center">
          <p className="font-mono text-xs text-muted-foreground">
            No SNR measurements recorded yet
          </p>
        </div>
      ) : (
        <div className="my-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {data.map((item, idx) => {
            const isHovered = hoveredIdx === idx;
            const hasVal = item.avgSnr !== null && item.sampleCount > 0;
            const snrVal = hasVal ? item.avgSnr! : null;

            const pct = snrVal !== null ? Math.min(100, Math.max(8, (snrVal / maxSnr) * 100)) : 0;

            return (
              <div
                key={item.label}
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
                className={`relative flex flex-col justify-between rounded-lg border p-3 transition-all ${
                  isHovered
                    ? "border-border-strong bg-surface/80 shadow-sm"
                    : "border-border bg-background/50"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-medium text-foreground">
                      {item.label}
                    </span>
                    <span className="font-mono text-[0.65rem] text-muted-foreground">
                      {item.sampleCount}x
                    </span>
                  </div>
                  <div className="mt-2 font-mono text-lg font-bold text-foreground">
                    {snrVal !== null ? `${snrVal.toFixed(1)}` : "—"}
                    {snrVal !== null && (
                      <span className="ml-1 text-xs font-normal text-muted-foreground">
                        dB
                      </span>
                    )}
                  </div>
                </div>

                <div className="mt-3">
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-border/60">
                    {hasVal && (
                      <div
                        className="h-full rounded-full bg-trace transition-all duration-700 ease-out"
                        style={{ width: `${pct}%` }}
                      />
                    )}
                  </div>
                </div>

                {isHovered && hasVal && (
                  <div className="absolute -top-7 left-1/2 -translate-x-1/2 z-10 whitespace-nowrap rounded border border-border-strong bg-background/95 px-2 py-0.5 font-mono text-[0.62rem] text-foreground shadow-sm">
                    Avg SNR: {snrVal?.toFixed(2)} dB
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
