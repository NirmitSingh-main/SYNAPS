import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

type Mode = "dark" | "light";

const ThemeContext = createContext<{ mode: Mode; toggle: () => void }>({
  mode: "dark",
  toggle: () => {},
});

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<Mode>("dark");

  useEffect(() => {
    const stored = window.localStorage.getItem("si-theme");
    if (stored === "light" || stored === "dark") setMode(stored);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", mode === "dark");
    window.localStorage.setItem("si-theme", mode);
  }, [mode]);

  const toggle = useCallback(() => setMode((m) => (m === "dark" ? "light" : "dark")), []);

  return <ThemeContext.Provider value={{ mode, toggle }}>{children}</ThemeContext.Provider>;
}

export const useTheme = () => useContext(ThemeContext);

export function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

/** Palette used by canvas layers (canvas can't read Tailwind tokens). */
export function palette(mode: Mode) {
  return mode === "dark"
    ? {
        particle: [138, 112, 88] as [number, number, number],
        particleActive: [148, 158, 174] as [number, number, number],
        trace: "rgba(178,148,116,0.75)",
        traceSoft: "rgba(150,160,176,0.35)",
        grid: "rgba(230,227,223,0.045)",
        spectro: [150, 160, 176] as [number, number, number],
      }
    : {
        particle: [140, 114, 88] as [number, number, number],
        particleActive: [110, 122, 138] as [number, number, number],
        trace: "rgba(112,84,58,0.7)",
        traceSoft: "rgba(104,116,132,0.4)",
        grid: "rgba(40,36,32,0.05)",
        spectro: [112, 84, 58] as [number, number, number],
      };
}
