import { useNavigate, useLocation } from "@tanstack/react-router";
import { useCallback, useEffect } from "react";

/**
 * Helper to smoothly scroll to a section by its DOM id.
 */
export function scrollToSection(sectionId: string): boolean {
  const targetId = sectionId.startsWith("#") ? sectionId.slice(1) : sectionId;
  const el = document.getElementById(targetId);
  if (el) {
    el.scrollIntoView({ behavior: "smooth" });
    return true;
  }
  return false;
}

/**
 * Reusable hook for unified SPA navigation across all pages.
 */
export function useAppNavigation() {
  const navigate = useNavigate();
  const location = useLocation();

  const navigateToHome = useCallback(
    (e?: React.MouseEvent) => {
      e?.preventDefault();
      if (location.pathname === "/") {
        window.scrollTo({ top: 0, behavior: "smooth" });
      } else {
        navigate({ to: "/" });
      }
    },
    [navigate, location.pathname]
  );

  const navigateToAnalyze = useCallback(
    (e?: React.MouseEvent) => {
      e?.preventDefault();
      if (location.pathname === "/analyze") {
        window.scrollTo({ top: 0, behavior: "smooth" });
      } else {
        navigate({ to: "/analyze" });
      }
    },
    [navigate, location.pathname]
  );

  const navigateToHowItWorks = useCallback(
    (e?: React.MouseEvent) => {
      e?.preventDefault();
      if (location.pathname === "/") {
        const scrolled = scrollToSection("how-it-works");
        if (scrolled) {
          window.history.pushState(null, "", "/#how-it-works");
        }
      } else {
        navigate({ to: "/", hash: "how-it-works" });
      }
    },
    [navigate, location.pathname]
  );

  return {
    navigateToHome,
    navigateToAnalyze,
    navigateToHowItWorks,
    pathname: location.pathname,
  };
}

/**
 * Hook to automatically handle hash-based scrolling on page render.
 * Ensures the DOM element is painted and ready before scrolling.
 */
export function useHashScroll() {
  const location = useLocation();

  useEffect(() => {
    const rawHash = window.location.hash || location.hash;
    if (!rawHash) return;

    const targetId = rawHash.startsWith("#") ? rawHash.slice(1) : rawHash;

    const tryScroll = () => {
      const el = document.getElementById(targetId);
      if (el) {
        el.scrollIntoView({ behavior: "smooth" });
        return true;
      }
      return false;
    };

    let timer: ReturnType<typeof setTimeout> | null = null;
    const rafId = requestAnimationFrame(() => {
      if (!tryScroll()) {
        timer = setTimeout(tryScroll, 100);
      }
    });

    return () => {
      cancelAnimationFrame(rafId);
      if (timer) {
        clearTimeout(timer);
      }
    };
  }, [location.pathname, location.hash]);
}
