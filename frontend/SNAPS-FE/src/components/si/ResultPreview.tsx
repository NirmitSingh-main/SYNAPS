import { useEffect, useRef } from "react";
import { palette, usePrefersReducedMotion, useTheme } from "@/lib/theme";
import { SignalTrace } from "./SignalTrace";

function useCanvas(draw: (ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => void, deps: unknown[]) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const reduced = usePrefersReducedMotion();
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    let w = 0;
    let h = 0;
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);
    const start = performance.now();
    const loop = (now: number) => {
      draw(ctx, w, h, reduced ? 2 : (now - start) / 1000);
      if (!reduced) raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced, ...deps]);
  return ref;
}

function Spectrum() {
  const { mode } = useTheme();
  const ref = useCanvas(
    (ctx, w, h, t) => {
      const pal = palette(mode);
      ctx.clearRect(0, 0, w, h);
      const bars = 46;
      const bw = w / bars;
      for (let i = 0; i < bars; i++) {
        const x = i / bars;
        const v =
          Math.exp(-Math.pow((x - 0.22) * 5, 2)) * (0.7 + 0.25 * Math.sin(t * 2 + i)) +
          Math.exp(-Math.pow((x - 0.62) * 8, 2)) * (0.5 + 0.3 * Math.sin(t * 3 + i * 0.6)) +
          0.05 * (0.5 + Math.sin(t * 5 + i * 2) * 0.5);
        const bh = Math.min(1, v) * h * 0.9;
        ctx.fillStyle = pal.trace;
        ctx.globalAlpha = 0.35 + Math.min(1, v) * 0.5;
        ctx.fillRect(i * bw + 1, h - bh, bw - 2, bh);
      }
      ctx.globalAlpha = 1;
    },
    [mode],
  );
  return <canvas ref={ref} className="h-24 w-full" aria-hidden />;
}

function Spectrogram() {
  const { mode } = useTheme();
  const ref = useCanvas(
    (ctx, w, h, t) => {
      const pal = palette(mode);
      ctx.clearRect(0, 0, w, h);
      const cols = 60;
      const rows = 14;
      const cw = w / cols;
      const ch = h / rows;
      for (let x = 0; x < cols; x++) {
        for (let y = 0; y < rows; y++) {
          const v =
            0.5 +
            0.5 *
              Math.sin(x * 0.35 + t * 1.2 + Math.cos(y * 0.5 + t * 0.4) * 1.6) *
              Math.exp(-Math.pow((y / rows - 0.45) * 2.4, 2));
          ctx.fillStyle = `rgba(${pal.spectro[0]},${pal.spectro[1]},${pal.spectro[2]},${Math.max(0, v) * 0.5})`;
          ctx.fillRect(x * cw, y * ch, cw - 0.5, ch - 0.5);
        }
      }
    },
    [mode],
  );
  return <canvas ref={ref} className="h-24 w-full" aria-hidden />;
}

export function ResultPreview() {
  return (
    <section id="preview" className="mx-auto max-w-6xl px-6 pb-32">
      <p className="label-mono">Analysis preview</p>
      <h2 className="mt-6 max-w-xl text-3xl tracking-[-0.02em] sm:text-4xl">
        What you get back after an analysis.
      </h2>

      <div className="mt-12 border border-border bg-background/60 backdrop-blur-sm">
        <div className="flex items-center justify-between border-b border-border px-6 py-4">
          <span className="text-sm">Signal analysis</span>
          <span className="font-mono text-[0.65rem] tracking-[0.14em] text-muted-foreground">ILLUSTRATIVE</span>
        </div>

        <div className="grid gap-px bg-border md:grid-cols-3">
          <div className="bg-background/60 p-6">
            <p className="label-mono">Detected signal</p>
            <p className="mt-4 border border-dashed border-border-strong px-4 py-3 text-sm text-muted-foreground">
              Signal classification
            </p>
            <div className="rule-line mt-6" />
            <dl className="mt-6 space-y-3 font-mono text-[0.7rem] tracking-[0.1em] text-muted-foreground">
              {["Signal type", "Sample rate", "Modulation", "Features"].map((k) => (
                <div key={k} className="flex items-center justify-between">
                  <dt>{k}</dt>
                  <dd className="text-border-strong">—</dd>
                </div>
              ))}
            </dl>
          </div>

          <div className="bg-background/60 p-6">
            <p className="label-mono">Waveform</p>
            <SignalTrace className="mt-4 h-24 w-full" />
            <div className="rule-line mt-6" />
            <p className="label-mono mt-4">Spectrogram</p>
            <Spectrogram />
          </div>

          <div className="bg-background/60 p-6">
            <p className="label-mono">Frequency spectrum</p>
            <Spectrum />
            <div className="rule-line mt-6" />
            <a
              href="#analyze"
              className="hover-arrow mt-6 inline-flex items-center gap-2 text-sm text-foreground transition-colors hover:text-signal"
            >
              View detailed analysis <span className="arrow font-mono text-signal">→</span>
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
