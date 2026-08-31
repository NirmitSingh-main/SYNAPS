import { useEffect, useRef } from "react";
import { palette, usePrefersReducedMotion, useTheme } from "@/lib/theme";

interface SpectrogramProps {
  rows: number[][]; // [timeRows][freqCols], values 0..1
  isDemoData?: boolean;
}

export function Spectrogram({ rows, isDemoData = true }: SpectrogramProps) {
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
          const v = Math.max(0, Math.min(1, rows[row]?.[col] ?? 0));
          // Restrained single-channel coloring — no rainbow
          // Uses the spectro palette color at variable opacity
          const alpha = v * 0.72;
          ctx.fillStyle = `rgba(${pal.spectro[0]},${pal.spectro[1]},${pal.spectro[2]},${alpha})`;
          ctx.fillRect(col * cw, row * ch, cw - 0.5, ch - 0.5);
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

  return (
    <div className="border border-border bg-background">
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
          <span className="font-mono text-[0.62rem] text-muted-foreground/60">t = 0</span>
          <span className="font-mono text-[0.62rem] text-muted-foreground/60">time × frequency</span>
          <span className="font-mono text-[0.62rem] text-muted-foreground/60">t = N</span>
        </div>
      </div>
    </div>
  );
}
