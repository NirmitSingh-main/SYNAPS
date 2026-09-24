import { useEffect, useRef } from "react";
import { palette, usePrefersReducedMotion, useTheme } from "@/lib/theme";

interface SpectrogramProps {
  rows: number[][];
  times?: number[];
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

function formatTime(seconds: number): string {
  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(1)} ms`;
  }

  return `${seconds.toFixed(2)} s`;
}

export function Spectrogram({
  rows,
  times,
  frequencies,
  isDemoData = true,
}: SpectrogramProps) {
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

      if (!rows || rows.length === 0) return;

      const numRows = rows.length;
      const numCols = rows[0]?.length ?? 0;

      if (numCols === 0) return;

      const cw = w / numCols;
      const ch = h / numRows;

      for (let row = 0; row < numRows; row++) {
        for (let col = 0; col < numCols; col++) {
          const v = Math.max(
            0,
            Math.min(1, rows[row]?.[col] ?? 0),
          );

          const alpha = v * 0.72;

          ctx.fillStyle = `rgba(${pal.spectro[0]},${pal.spectro[1]},${pal.spectro[2]},${alpha})`;

          ctx.fillRect(
            col * cw,
            row * ch,
            Math.max(0, cw - 0.5),
            Math.max(0, ch - 0.5),
          );
        }
      }

      if (!reduced) raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [mode, reduced, rows]);

  const startTime = times?.[0] ?? 0;

  const endTime =
    times?.[times.length - 1] ?? 0;

  const lowFreq = frequencies?.[0] ?? 0;

  const highFreq =
    frequencies?.[frequencies.length - 1] ?? 0;

  const frequencyLabel = frequencies
    ? `${formatFreq(lowFreq)} ↔ ${formatFreq(highFreq)}`
    : "time × frequency";

  return (
    <div className="floating-surface overflow-hidden">
      <div className="flex items-center justify-between border-b border-border px-6 py-4">
        <p className="label-mono">Spectrogram</p>

        {isDemoData && (
          <span className="font-mono text-[0.65rem] tracking-[0.14em] text-muted-foreground">
            ILLUSTRATIVE
          </span>
        )}
      </div>

      <div className="px-4 py-4">
        <canvas
          ref={canvasRef}
          className="h-36 w-full"
          role="img"
          aria-label="Spectrogram — signal energy across time and frequency"
        />

        <div className="mt-2 flex justify-between">
          <span className="font-mono text-[0.62rem] text-muted-foreground/60">
            {times
              ? `t = ${formatTime(startTime)}`
              : "t = 0"}
          </span>

          <span className="font-mono text-[0.62rem] text-muted-foreground/60">
            {frequencyLabel}
          </span>

          <span className="font-mono text-[0.62rem] text-muted-foreground/60">
            {times
              ? `t = ${formatTime(endTime)}`
              : "t = N"}
          </span>
        </div>
      </div>
    </div>
  );
}