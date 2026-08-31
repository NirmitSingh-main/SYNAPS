const items = [
  { label: "Automated", meta: "PIPELINE", copy: "Analysis begins with the signal." },
  { label: "Explainable", meta: "FEATURES", copy: "Insights are tied to extracted features." },
  { label: "Fast", meta: "THROUGHPUT", copy: "Move from signal to result efficiently." },
  { label: "Modular", meta: "ARCHITECTURE", copy: "Each processing stage can evolve independently." },
];

export function Capabilities() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-32">
      <p className="label-mono">Capabilities</p>
      <div className="mt-10 grid gap-px border border-border bg-border sm:grid-cols-2">
        {items.map((it, i) => (
          <div
            key={it.label}
            className="group relative bg-background/70 p-8 backdrop-blur-sm transition-colors duration-500 hover:bg-surface/80"
          >
            <div className="flex items-baseline justify-between">
              <h3 className="text-xl tracking-[-0.01em]">{it.label}</h3>
              <span className="font-mono text-[0.65rem] tracking-[0.14em] text-muted-foreground">
                {String(i + 1).padStart(2, "0")} · {it.meta}
              </span>
            </div>
            <div className="rule-line mt-5" />
            <p className="mt-5 max-w-[22ch] leading-relaxed text-muted-foreground">{it.copy}</p>
            <svg className="mt-8 h-6 w-full text-signal" viewBox="0 0 200 24" preserveAspectRatio="none">
              <path
                d="M0 12 H40 l6 -8 l6 16 l6 -12 H92 l5 -5 l5 10 l5 -5 H200"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
                opacity="0.5"
                className="transition-opacity duration-500 group-hover:opacity-100"
              />
            </svg>
          </div>
        ))}
      </div>
    </section>
  );
}
