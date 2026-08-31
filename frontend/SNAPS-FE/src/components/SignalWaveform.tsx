import { useEffect, useRef } from "react";
import { palette, usePrefersReducedMotion, useTheme } from "@/lib/theme";

interface SignalWaveformProps {
  samples: number[]; // Values in [-1, 1]
  label?: string;
  isDemoData?: boolean;
}

export function SignalWaveform({
  samples,
  label = "Waveform",
  isDemoData = true,
}: SignalWaveformProps) {
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

      if (!samples || samples.length === 0) return;

      // Center axis
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.strokeStyle = "var(--border)";
      ctx.lineWidth = 0.5;
      ctx.stroke();

      // Waveform
      ctx.beginPath();
      const amplitude = h * 0.4;
      for (let i = 0; i < samples.length; i++) {
        const x = (i / (samples.length - 1)) * w;
        const y = h / 2 - (samples[i] ?? 0) * amplitude;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = pal.trace;
      ctx.lineWidth = 1.25;
      ctx.lineJoin = "round";
      ctx.stroke();

      if (!reduced) raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [mode, reduced, samples]);

  return (
    <div className="border border-border bg-background">
      <div className="flex items-center justify-between border-b border-border px-6 py-4">
        <p className="label-mono">{label}</p>
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
          aria-label="Waveform visualization of signal amplitude over time"
        />
        {/* Axis labels */}
        <div className="mt-2 flex justify-between">
          <span className="font-mono text-[0.62rem] text-muted-foreground/60">t = 0</span>
          <span className="font-mono text-[0.62rem] text-muted-foreground/60">amplitude</span>
          <span className="font-mono text-[0.62rem] text-muted-foreground/60">t = N</span>
        </div>
      </div>
    </div>
  );
}
