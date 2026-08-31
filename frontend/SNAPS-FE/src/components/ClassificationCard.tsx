interface ClassificationCardProps {
  classification: string;
  confidence: number;
  isDemoData?: boolean;
}

export function ClassificationCard({
  classification,
  confidence,
  isDemoData = true,
}: ClassificationCardProps) {
  const pct = Math.round(confidence * 100);

  return (
    <div className="border border-border bg-background">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-6 py-4">
        <p className="label-mono">Signal classification</p>
        {isDemoData && (
          <span className="font-mono text-[0.65rem] tracking-[0.14em] text-muted-foreground">
            ILLUSTRATIVE
          </span>
        )}
      </div>

      {/* Classification result */}
      <div className="px-6 py-6">
        <p
          className="border border-dashed border-border-strong px-5 py-4 text-[1.05rem] text-foreground"
        >
          {classification}
        </p>

        <div className="rule-line mt-6" />

        {/* Confidence */}
        <div className="mt-6">
          <div className="flex items-center justify-between">
            <p className="label-mono">Confidence</p>
            <span className="font-mono text-[0.8rem] text-foreground">{pct}%</span>
          </div>
          {/* Thin confidence bar */}
          <div className="mt-3 h-px w-full bg-border">
            <div
              className="h-px bg-signal transition-[width] duration-1000 ease-out"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
