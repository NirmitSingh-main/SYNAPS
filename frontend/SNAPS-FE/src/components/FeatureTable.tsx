interface Feature {
  name: string;
  value: string;
  unit: string;
  description: string;
}

interface FeatureTableProps {
  features: Feature[];
  isDemoData?: boolean;
}

export function FeatureTable({ features, isDemoData = true }: FeatureTableProps) {
  return (
    <div className="border border-border bg-background">
      <div className="flex items-center justify-between border-b border-border px-6 py-4">
        <p className="label-mono">Extracted features</p>
        {isDemoData && (
          <span className="font-mono text-[0.65rem] tracking-[0.14em] text-muted-foreground">
            ILLUSTRATIVE
          </span>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[540px]" aria-label="Extracted signal features">
          <thead>
            <tr className="border-b border-border">
              <th className="px-6 py-3 text-left label-mono font-normal">Feature</th>
              <th className="px-4 py-3 text-right label-mono font-normal">Value</th>
              <th className="px-4 py-3 text-left label-mono font-normal">Unit</th>
              <th className="hidden px-4 py-3 text-left label-mono font-normal sm:table-cell">Description</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {features.map((f, i) => (
              <tr
                key={f.name}
                className="transition-colors duration-200 hover:bg-surface/60"
                style={{ opacity: 1 - i * 0.04 > 0.7 ? 1 : 0.82 }}
              >
                <td className="px-6 py-3 text-[0.8rem] text-foreground">{f.name}</td>
                <td className="px-4 py-3 text-right font-mono text-[0.8rem] text-foreground">
                  {f.value}
                </td>
                <td className="px-4 py-3 font-mono text-[0.72rem] text-muted-foreground">
                  {f.unit || "—"}
                </td>
                <td className="hidden px-4 py-3 text-[0.78rem] text-muted-foreground sm:table-cell">
                  {f.description}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
