import { useEffect, useRef } from "react";
import { palette, usePrefersReducedMotion, useTheme } from "@/lib/theme";

/** Thin animated signal trace used during processing to indicate activity. */
export function LoadingState() {
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

    const start = performance.now();

    const render = (now: number) => {
      const t = reduced ? 1.5 : (now - start) / 1000;
      const pal = palette(mode);

      ctx.clearRect(0, 0, w, h);

      // Primary signal trace
      ctx.beginPath();
      const steps = Math.max(120, Math.floor(w / 2));
      for (let i = 0; i <= steps; i++) {
        const x = i / steps;
        const v =
          Math.sin(x * Math.PI * 2 * 4 + t * 2.4) * 0.6 +
          Math.sin(x * Math.PI * 2 * 10 + t * 1.1) * 0.25 +
          Math.sin(x * Math.PI * 2 * 22 + t * 3.7) * 0.1;
        const y = h / 2 + v * h * 0.28;
        if (i === 0) ctx.moveTo(x * w, y);
        else ctx.lineTo(x * w, y);
      }
      ctx.strokeStyle = pal.trace;
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // Softer secondary trace
      ctx.beginPath();
      for (let i = 0; i <= steps; i++) {
        const x = i / steps;
        const v =
          Math.sin(x * Math.PI * 2 * 6 + t * 1.7 + 0.8) * 0.4 +
          Math.sin(x * Math.PI * 2 * 15 + t * 0.9) * 0.15;
        const y = h / 2 + v * h * 0.2;
        if (i === 0) ctx.moveTo(x * w, y);
        else ctx.lineTo(x * w, y);
      }
      ctx.strokeStyle = pal.traceSoft;
      ctx.lineWidth = 0.8;
      ctx.stroke();

      if (!reduced) raf = requestAnimationFrame(render);
    };

    raf = requestAnimationFrame(render);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [mode, reduced]);

  return (
    <canvas
      ref={canvasRef}
      className="h-16 w-full"
      aria-hidden
      aria-label="Signal processing animation"
    />
  );
}
