import { useEffect, useRef } from "react";
import { palette, usePrefersReducedMotion, useTheme } from "@/lib/theme";

interface FrequencySpectrumProps {
  bins: number[];
  frequencies?: number[];
  isDemoData?: boolean;
}

function formatFreq(hz: number): string {
  if (Math.abs(hz) >= 1_000_000) {
    return `${(hz / 1_000_000).toFixed(1)} MHz`;
  }

  if (Math.abs(hz) >= 1_000) {
    return `${(hz / 1_000).toFixed(0)} kHz`;
  }

  return `${hz.toFixed(0)} Hz`;
}

export function FrequencySpectrum({
  bins,
  frequencies,
  isDemoData = true,
}: FrequencySpectrumProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { mode } = useTheme();
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let w = 0;
    let h = 0;
    let raf = 0;

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

    const draw = () => {
      const pal = palette(mode);
      ctx.clearRect(0, 0, w, h);

      if (!bins || bins.length === 0) return;

      const bw = w / bins.length;

      for (let i = 0; i < bins.length; i++) {
        const mag = Math.min(
          1,
          Math.max(0, bins[i] ?? 0),
        );

        const bh = mag * h * 0.92;

        ctx.fillStyle = pal.trace;
        ctx.globalAlpha = 0.3 + mag * 0.55;

        ctx.fillRect(
          i * bw + 0.5,
          h - bh,
          Math.max(0, bw - 1),
          bh,
        );
      }

      ctx.globalAlpha = 1;

      ctx.beginPath();
      ctx.moveTo(0, h - 0.5);
      ctx.lineTo(w, h - 0.5);
      ctx.strokeStyle = "var(--border)";
      ctx.lineWidth = 0.5;
      ctx.stroke();

      if (!reduced) raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [mode, reduced, bins]);

  const first = frequencies?.[0] ?? 0;

  const middle =
    frequencies?.[
      Math.floor((frequencies.length - 1) / 2)
    ] ?? 0;

  const last =
    frequencies?.[frequencies.length - 1] ?? 0;

  return (
    <div className="border border-border bg-background">
      <div className="flex items-center justify-between border-b border-border px-6 py-4">
        <p className="label-mono">Frequency spectrum</p>

        {isDemoData && (
          <span className="font-mono text-[0.65rem] tracking-[0.14em] text-muted-foreground">
            ILLUSTRATIVE
          </span>
        )}
      </div>

      <div className="px-4 py-4">
        <canvas
          ref={canvasRef}
          className="h-32 w-full"
          role="img"
          aria-label="Frequency spectrum — normalized power versus frequency"
        />

        <div className="mt-2 flex justify-between">
          <span className="font-mono text-[0.62rem] text-muted-foreground/60">
            {frequencies ? formatFreq(first) : "0 Hz"}
          </span>

          <span className="font-mono text-[0.62rem] text-muted-foreground/60">
            {frequencies ? formatFreq(middle) : "power"}
          </span>

          <span className="font-mono text-[0.62rem] text-muted-foreground/60">
            {frequencies ? formatFreq(last) : "fs/2"}
          </span>
        </div>
      </div>
    </div>
  );
}