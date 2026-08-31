const STAGES = [
  { n: "01", label: "Input",       copy: "Signal file received and loaded." },
  { n: "02", label: "Detect",      copy: "Identifying signal structure and boundaries." },
  { n: "03", label: "Preprocess",  copy: "Normalising and preparing for analysis." },
  { n: "04", label: "DSP",         copy: "Applying digital signal processing transforms." },
  { n: "05", label: "Synchronize", copy: "Aligning signal phase and timing references." },
  { n: "06", label: "Features",    copy: "Extracting measurable signal characteristics." },
  { n: "07", label: "AI",          copy: "Running classification model on extracted features." },
  { n: "08", label: "Result",      copy: "Analysis complete. Preparing output." },
];

interface AnalysisPipelineProps {
  activeStage: number; // 0-indexed
  completed?: boolean;
}

export function AnalysisPipeline({ activeStage, completed = false }: AnalysisPipelineProps) {
  const progress = completed ? 1 : activeStage / (STAGES.length - 1);

  return (
    <div className="w-full">
      {/* ---- Desktop: horizontal timeline ---- */}
      <div className="hidden md:block">
        <div className="relative">
          {/* Track */}
          <div className="pointer-events-none absolute top-[11px] right-0 left-0 h-px bg-border" />
          {/* Filled track */}
          <div
            className="pointer-events-none absolute top-[11px] left-0 h-px bg-signal/70 transition-[width] duration-700 ease-out"
            style={{ width: `${progress * 100}%` }}
          />
          {/* Moving dot */}
          <div
            className="pointer-events-none absolute top-[8px] h-[7px] w-[7px] rounded-full bg-signal transition-[left] duration-700 ease-out"
            style={{ left: `calc(${progress * 100}% - 3.5px)` }}
          />

          {/* Stage nodes */}
          <div className="relative flex items-start justify-between">
            {STAGES.map((s, i) => {
              const isActive = i === activeStage && !completed;
              const isDone = i < activeStage || completed;

              return (
                <div key={s.n} className="flex flex-col items-center" style={{ minWidth: 0 }}>
                  {/* Number above line */}
                  <span
                    className="font-mono text-[0.65rem] tracking-[0.14em] transition-colors duration-500"
                    style={{
                      color: isActive ? "var(--signal)" : isDone ? "var(--muted-foreground)" : "var(--border-strong)",
                    }}
                  >
                    {s.n}
                  </span>

                  {/* Spacer (line height) */}
                  <div className="h-4" />

                  {/* Label below line */}
                  <span
                    className="whitespace-nowrap text-[0.75rem] transition-all duration-500"
                    style={{
                      color: isActive ? "var(--foreground)" : isDone ? "var(--muted-foreground)" : "var(--border-strong)",
                      opacity: isActive ? 1 : isDone ? 0.6 : 0.35,
                    }}
                  >
                    {s.label}
                  </span>
                  {/* Active underline */}
                  <span
                    className="mt-1 h-px bg-signal transition-all duration-500"
                    style={{ width: isActive ? "20px" : "0px" }}
                  />
                </div>
              );
            })}
          </div>
        </div>

        {/* Active stage explanation */}
        {!completed && (
          <div
            key={activeStage}
            className="mt-12 animate-fade-in border-l border-signal pl-6"
          >
            <p className="font-mono text-[0.68rem] tracking-[0.16em] text-signal">
              {STAGES[activeStage]?.n}
            </p>
            <h3 className="mt-3 text-xl tracking-[-0.02em]">
              {STAGES[activeStage]?.label}
            </h3>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              {STAGES[activeStage]?.copy}
            </p>
          </div>
        )}

        {completed && (
          <div className="mt-12 border-l border-signal pl-6">
            <p className="font-mono text-[0.68rem] tracking-[0.16em] text-signal">08</p>
            <h3 className="mt-3 text-xl tracking-[-0.02em]">Complete</h3>
            <p className="mt-2 text-sm text-muted-foreground">Preparing results…</p>
          </div>
        )}
      </div>

      {/* ---- Mobile: vertical list ---- */}
      <div className="md:hidden">
        {/* Progress bar */}
        <div className="relative mb-8 h-px w-full bg-border">
          <div
            className="absolute top-0 left-0 h-px bg-signal transition-[width] duration-700 ease-out"
            style={{ width: `${progress * 100}%` }}
          />
        </div>

        <ol className="space-y-4">
          {STAGES.map((s, i) => {
            const isActive = i === activeStage && !completed;
            const isDone = i < activeStage || completed;

            return (
              <li
                key={s.n}
                className="flex items-start gap-4 transition-opacity duration-500"
                style={{ opacity: isActive ? 1 : isDone ? 0.55 : 0.28 }}
              >
                <span className="font-mono text-[0.68rem] tracking-[0.14em] text-signal pt-0.5">
                  {s.n}
                </span>
                <div>
                  <span
                    className="text-[0.85rem]"
                    style={{ color: isActive ? "var(--foreground)" : "var(--muted-foreground)" }}
                  >
                    {s.label}
                  </span>
                  {isActive && (
                    <p className="mt-0.5 text-[0.75rem] leading-relaxed text-muted-foreground">
                      {s.copy}
                    </p>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
