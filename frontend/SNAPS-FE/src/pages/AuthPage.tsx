import { useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { ThemeProvider } from "@/lib/theme";
import { useAuth } from "@/lib/AuthContext";
import { Lock, Mail, ArrowRight, AlertCircle, CheckCircle } from "lucide-react";

export function AuthPage() {
  const { signIn, signUp } = useAuth();
  const navigate = useNavigate();

  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    setLoading(true);

    try {
      if (mode === "signin") {
        await signIn(email, password);
        navigate({ to: "/analyze" });
      } else {
        await signUp(email, password);
        setSuccess(
          "Account created. Check your email for a confirmation link, then sign in."
        );
        setMode("signin");
      }
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Authentication failed.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <ThemeProvider>
      <div className="relative flex min-h-screen flex-col items-center justify-center bg-background px-6">
        {/* Subtle grid background */}
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `repeating-linear-gradient(0deg, transparent, transparent 39px, var(--color-border-strong) 39px, var(--color-border-strong) 40px), repeating-linear-gradient(90deg, transparent, transparent 39px, var(--color-border-strong) 39px, var(--color-border-strong) 40px)`,
          }}
        />

        {/* Back to home */}
        <a
          href="/"
          className="group absolute left-6 top-6 flex items-center gap-2 font-mono text-[0.72rem] tracking-widest text-muted-foreground transition-colors hover:text-foreground md:left-10 md:top-8"
        >
          <span className="transition-transform group-hover:-translate-x-0.5">←</span>
          HOME
        </a>

        <div className="relative z-10 w-full max-w-sm">
          {/* Brand */}
          <div className="mb-10 text-center">
            <p className="label-mono mb-3">Signal Intelligence</p>
            <h1 className="text-2xl tracking-[-0.02em]">
              {mode === "signin" ? "Sign in" : "Create account"}
            </h1>
          </div>

          {/* Card */}
          <div
            className="border border-border bg-surface/40 backdrop-blur-sm"
            style={{ boxShadow: "0 4px 32px rgba(0,0,0,0.18)" }}
          >
            {/* Tab toggle */}
            <div className="grid grid-cols-2 border-b border-border">
              <button
                type="button"
                onClick={() => { setMode("signin"); setError(null); setSuccess(null); }}
                className={`py-3 font-mono text-[0.72rem] tracking-widest uppercase transition-colors ${mode === "signin" ? "bg-background/60 text-foreground" : "text-muted-foreground hover:text-foreground"}`}
              >
                Sign In
              </button>
              <button
                type="button"
                onClick={() => { setMode("signup"); setError(null); setSuccess(null); }}
                className={`py-3 font-mono text-[0.72rem] tracking-widest uppercase transition-colors border-l border-border ${mode === "signup" ? "bg-background/60 text-foreground" : "text-muted-foreground hover:text-foreground"}`}
              >
                Create Account
              </button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-0 p-6">
              {/* Email */}
              <div className="mb-4">
                <label className="label-mono mb-2 block" htmlFor="auth-email">
                  Email
                </label>
                <div className="relative">
                  <Mail className="pointer-events-none absolute left-3 top-1/2 h-[14px] w-[14px] -translate-y-1/2 text-muted-foreground" />
                  <input
                    id="auth-email"
                    type="email"
                    required
                    autoComplete="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    className="w-full border border-border bg-background/60 py-2.5 pl-9 pr-3 font-mono text-[0.82rem] text-foreground placeholder:text-muted-foreground/50 outline-none focus:border-border-strong transition-colors"
                  />
                </div>
              </div>

              {/* Password */}
              <div className="mb-6">
                <label className="label-mono mb-2 block" htmlFor="auth-password">
                  Password
                </label>
                <div className="relative">
                  <Lock className="pointer-events-none absolute left-3 top-1/2 h-[14px] w-[14px] -translate-y-1/2 text-muted-foreground" />
                  <input
                    id="auth-password"
                    type="password"
                    required
                    autoComplete={mode === "signin" ? "current-password" : "new-password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full border border-border bg-background/60 py-2.5 pl-9 pr-3 font-mono text-[0.82rem] text-foreground placeholder:text-muted-foreground/50 outline-none focus:border-border-strong transition-colors"
                  />
                </div>
              </div>

              {/* Error */}
              {error && (
                <div className="mb-4 flex items-start gap-2 border border-destructive/40 bg-destructive/10 px-3 py-2.5 text-[0.75rem] text-destructive">
                  <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              {/* Success */}
              {success && (
                <div className="mb-4 flex items-start gap-2 border border-green-500/40 bg-green-500/10 px-3 py-2.5 text-[0.75rem] text-green-400">
                  <CheckCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>{success}</span>
                </div>
              )}

              {/* Submit */}
              <button
                type="submit"
                disabled={loading}
                className="hover-arrow inline-flex w-full items-center justify-between border border-border-strong bg-foreground/5 px-5 py-3 font-mono text-[0.82rem] text-foreground transition-colors hover:border-signal hover:bg-signal/10 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <span>
                  {loading
                    ? "Processing..."
                    : mode === "signin"
                    ? "Sign In"
                    : "Create Account"}
                </span>
                <ArrowRight className="arrow h-4 w-4 text-signal" />
              </button>
            </form>
          </div>

          <p className="mt-6 text-center font-mono text-[0.68rem] text-muted-foreground">
            {mode === "signin" ? (
              <>
                No account?{" "}
                <button
                  type="button"
                  onClick={() => { setMode("signup"); setError(null); }}
                  className="text-foreground underline-offset-2 hover:underline"
                >
                  Create one
                </button>
              </>
            ) : (
              <>
                Already have an account?{" "}
                <button
                  type="button"
                  onClick={() => { setMode("signin"); setError(null); }}
                  className="text-foreground underline-offset-2 hover:underline"
                >
                  Sign in
                </button>
              </>
            )}
          </p>
        </div>
      </div>
    </ThemeProvider>
  );
}
