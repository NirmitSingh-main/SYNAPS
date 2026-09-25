interface ClassificationCardProps {
  classification: string;
  confidence: number;
  isDemoData?: boolean | undefined;
  signalType?: string | undefined;
  detectedComponents?: string[] | undefined;
}

export function ClassificationCard({
  classification,
  confidence,
  isDemoData = true,
  signalType,
  detectedComponents,
}: ClassificationCardProps) {
  const pct = Math.round(confidence * 100);
  const isMixed = classification === "MIXED" || signalType === "MIXED";

  return (
    <div className="floating-surface overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-6 py-4">
        <p className="label-mono">
          {isMixed ? "Signal classification · Multi-carrier" : "Signal classification"}
        </p>
        {isDemoData && (
          <span className="font-mono text-[0.65rem] tracking-[0.14em] text-muted-foreground">
            ILLUSTRATIVE
          </span>
        )}
      </div>

      {/* Classification result */}
      <div className="px-6 py-6">
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-dashed border-border-strong px-5 py-4">
          <p className="text-[1.05rem] font-medium text-foreground">
            {classification}
          </p>
          {isMixed && detectedComponents && detectedComponents.length > 0 && (
            <div className="flex items-center gap-1.5 font-mono text-[0.72rem] text-signal">
              <span className="text-muted-foreground">Comps:</span>
              <span>{detectedComponents.join(" + ")}</span>
            </div>
          )}
        </div>

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
