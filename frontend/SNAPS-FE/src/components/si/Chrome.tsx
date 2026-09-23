import { useRef, useState, type ReactNode } from "react";
import { Link } from "@tanstack/react-router";
import { Moon, Sun } from "lucide-react";
import { useTheme, usePrefersReducedMotion } from "@/lib/theme";
import { useAppNavigation } from "@/lib/navigation";

export function MagneticLink({
  children,
  href,
  variant = "primary",
  onClick,
}: {
  children: ReactNode;
  href: string;
  variant?: "primary" | "ghost";
  onClick?: (e: React.MouseEvent) => void;
}) {
  const ref = useRef<HTMLAnchorElement | null>(null);
  const reduced = usePrefersReducedMotion();
  const [t, setT] = useState({ x: 0, y: 0 });

  const onMove = (e: React.MouseEvent) => {
    if (reduced || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    setT({ x: (e.clientX - (r.left + r.width / 2)) * 0.06, y: (e.clientY - (r.top + r.height / 2)) * 0.12 });
  };

  const base =
    "group relative inline-flex items-center justify-between gap-8 border px-6 py-4 text-[0.95rem] transition-colors duration-300 hover-arrow";
  const styles =
    variant === "primary"
      ? "border-border-strong bg-surface/60 text-foreground hover:border-signal hover:bg-secondary/70 backdrop-blur-sm"
      : "border-transparent px-0 py-2 text-muted-foreground hover:text-foreground";

  const isInternal = href.startsWith("/") && !href.includes("#");
  const sharedProps = {
    onMouseMove: onMove,
    onMouseLeave: () => setT({ x: 0, y: 0 }),
    style: { transform: `translate3d(${t.x}px, ${t.y}px, 0)` },
    className: `${base} ${styles}`,
    onClick,
  };

  const inner = (
    <>
      <span>{children}</span>
      <span className="arrow font-mono text-signal">→</span>
    </>
  );

  return isInternal ? (
    <Link ref={ref as React.Ref<HTMLAnchorElement>} to={href} {...sharedProps}>
      {inner}
    </Link>
  ) : (
    <a ref={ref} href={href} {...sharedProps}>
      {inner}
    </a>
  );
}

export function Header() {
  const { mode, toggle } = useTheme();
  const { navigateToHome, navigateToAnalyze, navigateToHowItWorks } = useAppNavigation();

  const links = [
    { label: "Home", href: "/", onClick: navigateToHome },
    { label: "Analyze", href: "/analyze", onClick: navigateToAnalyze },
    { label: "How it works", href: "/#how-it-works", onClick: navigateToHowItWorks },
  ];

  return (
    <header className="fixed top-0 left-0 z-50 w-full border-b border-border/60 bg-background/70 backdrop-blur-md transition-colors duration-500">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
        <a
          href="/"
          onClick={navigateToHome}
          className="text-[0.95rem] tracking-tight text-foreground"
        >
          Signal Intelligence
        </a>
        <nav className="hidden items-center gap-8 md:flex">
          {links.map((l) => (
            <a
              key={l.label}
              href={l.href}
              onClick={l.onClick}
              className="text-[0.85rem] text-muted-foreground transition-colors hover:text-foreground"
            >
              {l.label}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-4">
          <button
            onClick={toggle}
            aria-label={mode === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            className="inline-flex h-8 w-8 items-center justify-center rounded-sm text-muted-foreground transition-colors hover:text-foreground"
          >
            {mode === "dark" ? <Sun className="h-[15px] w-[15px]" /> : <Moon className="h-[15px] w-[15px]" />}
          </button>
          <a
            href="/analyze"
            onClick={navigateToAnalyze}
            className="hover-arrow inline-flex items-center gap-2 border border-border px-3 py-1.5 text-[0.8rem] text-foreground transition-colors hover:border-border-strong hover:bg-secondary/60"
          >
            Analyze <span className="arrow font-mono text-signal">→</span>
          </a>
        </div>
      </div>
    </header>
  );
}

export function Footer() {
  const { navigateToHowItWorks } = useAppNavigation();

  return (
    <footer className="border-t border-border/70">
      <div className="mx-auto max-w-6xl px-6 py-20">
        <div className="flex flex-col justify-between gap-12 md:flex-row">
          <div className="max-w-sm">
            <p className="text-[0.95rem] text-foreground">Signal Intelligence</p>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              Automated signal analysis and AI-powered characterization.
            </p>
          </div>
          <div className="flex flex-col gap-3">
            <a
              href="mailto:pratyushh0212@gmail.com"
              className="hover-arrow inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              Reach Us <span className="arrow font-mono text-signal">→</span>
            </a>
            <a
              href="mailto:pratyushh0212@gmail.com?subject=SYNAPS%20Feedback"
              className="hover-arrow inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              Feedback <span className="arrow font-mono text-signal">→</span>
            </a>
            <a
              href="https://github.com/NirmitSingh-main/SYNAPS.git"
              target="_blank"
              rel="noopener noreferrer"
              className="hover-arrow inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              GitHub <span className="arrow font-mono text-signal">→</span>
            </a>
            <a
              href="/#how-it-works"
              onClick={navigateToHowItWorks}
              className="hover-arrow inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              How it works <span className="arrow font-mono text-signal">→</span>
            </a>
          </div>
        </div>
        <div className="rule-line mt-16" />
        <div className="mt-6 flex flex-col gap-2 text-xs text-muted-foreground md:flex-row md:items-center md:justify-between">
          <span className="font-mono">© 2026 Signal Intelligence</span>
          <span>Visualizations on this page are illustrative and not generated from a real analysis.</span>
        </div>
      </div>
    </footer>
  );
}

