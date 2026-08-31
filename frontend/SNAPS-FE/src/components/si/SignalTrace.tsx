import { useEffect, useRef } from "react";
import { palette, usePrefersReducedMotion, useTheme } from "@/lib/theme";

type Kind = 0 | 1 | 2 | 3 | 4;

function sample(kind: Kind, x: number, t: number): number {
  switch (kind) {
    case 0: // smooth oscillation
      return Math.sin(x * 6 + t * 1.2) * 0.7;
    case 1: // modulated waveform
      return Math.sin(x * 22 + t * 3) * (0.35 + 0.5 * Math.sin(x * 3 + t));
    case 2: // stepped signal
      return (Math.round(Math.sin(x * 5 + t) * 3) / 3) * 0.75;
    case 3: // noisy trace
      return (
        Math.sin(x * 9 + t * 1.5) * 0.45 +
        Math.sin(x * 61 + t * 9) * 0.14 +
        Math.sin(x * 133 + t * 4) * 0.07
      );
    case 4: // spectral envelope
      return (
        Math.exp(-Math.pow((x - 0.3) * 6, 2)) * Math.sin(x * 40 + t * 2) * 0.8 +
        Math.exp(-Math.pow((x - 0.72) * 9, 2)) * Math.sin(x * 70 + t * 3) * 0.6
      );
  }
}

export function SignalTrace({ className = "" }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const { mode } = useTheme();
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const canvas = ref.current;
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

    const MORPH = 7; // seconds per stage
    const start = performance.now();

    const render = (now: number) => {
      const elapsed = (now - start) / 1000;
      const t = reduced ? 3 : elapsed;
      const phase = (t / MORPH) % 5;
      const from = Math.floor(phase) as Kind;
      const to = ((from + 1) % 5) as Kind;
      const raw = phase - Math.floor(phase);
      const k = raw * raw * (3 - 2 * raw);

      const pal = palette(modeRef.current);
      ctx.clearRect(0, 0, w, h);

      const drawPath = (amp: number, yOff: number, style: string, width: number) => {
        ctx.beginPath();
        const steps = Math.max(120, Math.floor(w / 2));
        for (let i = 0; i <= steps; i++) {
          const x = i / steps;
          const v = sample(from, x, t) * (1 - k) + sample(to, x, t) * k;
          const y = h / 2 + yOff + v * amp;
          if (i === 0) ctx.moveTo(x * w, y);
          else ctx.lineTo(x * w, y);
        }
        ctx.strokeStyle = style;
        ctx.lineWidth = width;
        ctx.stroke();
      };

      drawPath(h * 0.3, 0, pal.trace, 1.4);
      drawPath(h * 0.18, h * 0.13, pal.traceSoft, 1);
      drawPath(h * 0.1, -h * 0.16, pal.traceSoft, 0.8);

      if (!reduced) raf = requestAnimationFrame(render);
    };

    raf = requestAnimationFrame(render);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [reduced, mode]);

  return <canvas ref={ref} className={className} aria-hidden />;
}
