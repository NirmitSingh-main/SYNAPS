import { SignalTrace } from "./SignalTrace";
import { MagneticLink } from "./Chrome";

const tags = ["WAV", "IQ", "DSP", "AI insights"];

export function Hero() {
  return (
    <section id="top" className="relative flex min-h-[100svh] items-center pt-16">
      <div className="pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2">
        <SignalTrace className="h-[44vh] w-full opacity-60" />
      </div>

      <div className="relative mx-auto w-full max-w-6xl px-6">
        <div className="grid gap-12 md:grid-cols-12">
          <div className="hidden md:col-span-2 md:block">
            <p className="text-[0.7rem] tracking-[0.2em] text-muted-foreground/80 uppercase">
              SI <span className="text-muted-foreground/50">/</span>{" "}
              <span className="font-mono text-[0.68rem] tracking-normal">01</span>
            </p>
            <div className="rule-line mt-4 w-2/3" />
            <p className="mt-4 text-[0.72rem] leading-[1.5] tracking-[0.18em] text-muted-foreground/70 uppercase">
              Signal
              <br />
              Observatory
            </p>
          </div>

          <div className="md:col-span-10">
            <p className="text-[0.78rem] tracking-[0.1em] text-muted-foreground">Signal Intelligence</p>
            <h1 className="mt-8 max-w-3xl text-[2.6rem] leading-[1.04] tracking-[-0.03em] text-foreground sm:text-6xl lg:text-[4.5rem]">
              Turn signals into
              <br />
              <span className="text-muted-foreground">intelligence.</span>
            </h1>
            <p className="mt-8 max-w-md text-base leading-relaxed text-muted-foreground">
              Automated signal analysis, characterization and AI-powered classification.
            </p>

            <div className="mt-12 flex flex-col items-start gap-6 sm:flex-row sm:items-center">
              <MagneticLink href="/analyze">Analyze a signal</MagneticLink>
              <MagneticLink href="#how-it-works" variant="ghost">
                How it works
              </MagneticLink>
            </div>

            <div className="mt-16 flex flex-wrap items-center gap-2.5">
              {tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex cursor-default items-center rounded-full border border-border bg-surface/40 px-4 py-1.5 text-[0.75rem] tracking-[0.04em] text-muted-foreground transition-colors duration-300 hover:border-border-strong hover:bg-surface/70 hover:text-foreground"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
