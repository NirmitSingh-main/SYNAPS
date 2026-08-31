interface Metric {
  label: string;
  value: string;
  unit?: string;
}

interface SignalMetricsProps {
  metrics: Metric[];
  isDemoData?: boolean;
}

export function SignalMetrics({ metrics, isDemoData = true }: SignalMetricsProps) {
  return (
    <div className="border border-border bg-background">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-6 py-4">
        <p className="label-mono">Signal metrics</p>
        {isDemoData && (
          <span className="font-mono text-[0.65rem] tracking-[0.14em] text-muted-foreground">
            ILLUSTRATIVE
          </span>
        )}
      </div>

      {/* Metrics list */}
      <dl className="divide-y divide-border">
        {metrics.map((m) => (
          <div
            key={m.label}
            className="flex items-center justify-between px-6 py-3.5"
          >
            <dt className="text-[0.8rem] text-muted-foreground">{m.label}</dt>
            <dd className="font-mono text-[0.8rem] text-foreground">
              {m.value}
              {m.unit && (
                <span className="ml-1 text-[0.72rem] text-muted-foreground">
                  {m.unit}
                </span>
              )}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
