import { useEffect, useRef } from "react";
import { palette, usePrefersReducedMotion, useTheme } from "@/lib/theme";

type P = { x: number; y: number; bx: number; by: number; vx: number; vy: number; a: number; k: number };

export function ParticleField() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { mode } = useTheme();
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let w = 0;
    let h = 0;
    let dpr = 1;
    let particles: P[] = [];
    let raf = 0;
    const mouse = { x: -9999, y: -9999, tx: -9999, ty: -9999, active: false };

    const isMobile = () => window.innerWidth < 768;

    const build = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const density = isMobile() ? 26000 : 13000;
      const count = Math.min(isMobile() ? 45 : 150, Math.round((w * h) / density));
      particles = Array.from({ length: count }, () => {
        const x = Math.random() * w;
        const y = Math.random() * h;
        return { x, y, bx: x, by: y, vx: 0, vy: 0, a: 0.25 + Math.random() * 0.4, k: 0 };
      });
    };

    build();

    const onResize = () => build();
    const onMove = (e: PointerEvent) => {
      mouse.tx = e.clientX;
      mouse.ty = e.clientY - canvas.getBoundingClientRect().top;
      mouse.active = true;
    };
    const onLeave = () => {
      mouse.active = false;
      mouse.tx = -9999;
      mouse.ty = -9999;
    };

    window.addEventListener("resize", onResize);
    if (!isMobile()) {
      window.addEventListener("pointermove", onMove, { passive: true });
      window.addEventListener("pointerleave", onLeave);
    }

    let t = 0;
    const R = 150;

    const draw = () => {
      t += 0.004;
      const pal = palette(modeRef.current);
      ctx.clearRect(0, 0, w, h);

      mouse.x += (mouse.tx - mouse.x) * 0.12;
      mouse.y += (mouse.ty - mouse.y) * 0.12;

      for (const p of particles) {
        // ambient drift
        const dx = Math.sin(t + p.bx * 0.01) * 6;
        const dy = Math.cos(t * 0.8 + p.by * 0.012) * 6;
        let tx = p.bx + dx;
        let ty = p.by + dy;

        const mx = p.x - mouse.x;
        const my = p.y - mouse.y;
        const d = Math.hypot(mx, my);
        let k = 0;
        if (d < R) {
          k = 1 - d / R;
          const push = k * k * 26;
          tx += (mx / (d || 1)) * push;
          ty += (my / (d || 1)) * push;
        }
        p.k += (k - p.k) * 0.06;
        p.x += (tx - p.x) * 0.06;
        p.y += (ty - p.y) * 0.06;
      }

      // connections
      for (let i = 0; i < particles.length; i++) {
        const a = particles[i]!;
        for (let j = i + 1; j < particles.length; j++) {
          const b = particles[j]!;
          const dd = Math.hypot(a.x - b.x, a.y - b.y);
          if (dd > 108) continue;
          const near = Math.max(a.k, b.k);
          const alpha = (1 - dd / 108) * (0.05 + near * 0.22);
          const c = mix(pal.particle, pal.particleActive, near);
          ctx.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
          ctx.lineWidth = 0.6;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }

      for (const p of particles) {
        const c = mix(pal.particle, pal.particleActive, p.k);
        ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${p.a * (0.5 + p.k * 0.9)})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1 + p.k * 0.9, 0, Math.PI * 2);
        ctx.fill();
      }

      raf = requestAnimationFrame(draw);
    };

    if (reduced) {
      // single static frame
      const pal = palette(modeRef.current);
      ctx.clearRect(0, 0, w, h);
      for (const p of particles) {
        ctx.fillStyle = `rgba(${pal.particle[0]},${pal.particle[1]},${pal.particle[2]},${p.a * 0.5})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1, 0, Math.PI * 2);
        ctx.fill();
      }
    } else {
      raf = requestAnimationFrame(draw);
    }

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerleave", onLeave);
    };
  }, [reduced, mode]);

  return (
    <div className="pointer-events-none fixed inset-0 -z-10">
      <GridLayer />
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" aria-hidden />
      <div className="absolute inset-0 opacity-70" style={{ background: "radial-gradient(ellipse at 50% 0%, transparent 35%, var(--background) 100%)" }} />
    </div>
  );
}

function GridLayer() {
  const { mode } = useTheme();
  const line = mode === "dark" ? "rgba(230,227,223,0.045)" : "rgba(40,36,32,0.055)";
  return (
    <div
      className="absolute inset-0 transition-opacity duration-700"
      style={{
        backgroundImage: `linear-gradient(to right, ${line} 1px, transparent 1px), linear-gradient(to bottom, ${line} 1px, transparent 1px)`,
        backgroundSize: "72px 72px",
      }}
    />
  );
}

function mix(a: [number, number, number], b: [number, number, number], k: number) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * k),
    Math.round(a[1] + (b[1] - a[1]) * k),
    Math.round(a[2] + (b[2] - a[2]) * k),
  ] as [number, number, number];
}
