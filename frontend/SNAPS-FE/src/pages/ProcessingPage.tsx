import { useEffect, useState, useRef } from "react";
import { useNavigate } from "@tanstack/react-router";
import { ThemeProvider } from "@/lib/theme";
import { AnalysisPipeline } from "@/components/AnalysisPipeline";
import { LoadingState } from "@/components/LoadingState";
import {
  getAnalysisFile,
  getAnalysisSample,
  setAnalysisData,
  setAnalysisError,
} from "@/lib/analysisStore";
import {
  uploadAndAnalyzeSignal,
  analyzeDatasetSample,
  AnalysisResponse,
} from "@/services/api";

const STAGE_INTERVAL_MS = 600;
const TOTAL_STAGES = 8;

export function ProcessingPage() {
  const [activeStage, setActiveStage] = useState(0);
  const [completed, setCompleted] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const navigate = useNavigate();

  const file = getAnalysisFile();
  const sampleId = getAnalysisSample();

  const apiResultRef = useRef<AnalysisResponse | null>(null);
  const animationFinishedRef = useRef(false);

  // If arrived without file or sampleId, return to analyze page
  useEffect(() => {
    if (!file && !sampleId) {
      navigate({ to: "/analyze" });
    }
  }, [file, sampleId, navigate]);

  // Execute Real Backend Analysis
  useEffect(() => {
    if (!file && !sampleId) return;

    let isMounted = true;

    async function runBackendAnalysis() {
      try {
        let response: AnalysisResponse;
        if (file) {
          response = await uploadAndAnalyzeSignal(file);
        } else if (sampleId) {
          response = await analyzeDatasetSample(sampleId);
        } else {
          throw new Error("No input file or sample selected.");
        }

        if (isMounted) {
          apiResultRef.current = response;
          setAnalysisData(response);

          // If stage animation already reached the end, navigate immediately
          if (animationFinishedRef.current) {
            setCompleted(true);
            setTimeout(() => navigate({ to: "/results" }), 500);
          }
        }
      } catch (err: any) {
        console.error("Analysis pipeline error:", err);
        if (isMounted) {
          setErrorMsg(err?.message || "Failed to communicate with AI DSP backend.");
          setAnalysisError(err?.message || "Analysis failure.");
        }
      }
    }

    runBackendAnalysis();

    return () => {
      isMounted = false;
    };
  }, [file, sampleId, navigate]);

  // Smooth stage progression animation
  useEffect(() => {
    let stage = 0;
    const interval = setInterval(() => {
      stage += 1;
      if (stage < TOTAL_STAGES - 1) {
        setActiveStage(stage);
      } else {
        clearInterval(interval);
        setActiveStage(TOTAL_STAGES - 1);
        animationFinishedRef.current = true;

        // If backend API result is already ready, complete immediately
        if (apiResultRef.current) {
          setCompleted(true);
          setTimeout(() => {
            navigate({ to: "/results" });
          }, 600);
        }
      }
    }, STAGE_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [navigate]);

  return (
    <ThemeProvider>
      <div className="relative min-h-screen">
        <main className="mx-auto max-w-3xl px-6 pb-32 pt-32">
          {/* Page heading */}
          <div className="mb-4">
            <p className="label-mono">Master Engine Pipeline</p>
            <h1 className="mt-4 text-3xl tracking-[-0.02em] sm:text-4xl">
              Executing Signal Analysis
            </h1>
          </div>

          <p className="mb-10 text-sm text-muted-foreground font-mono">
            {errorMsg ? (
              <span className="text-red-400">Error: {errorMsg}</span>
            ) : (
              <span>Running live multi-stage DSP spectral estimation, synchronization, neural Transformer inference, and RF fingerprint extraction.</span>
            )}
          </p>

          <div className="rule-line mb-12" />

          {/* Animated signal activity indicator */}
          <div className="mb-12 border border-border px-4 py-6">
            <p className="label-mono mb-4">Signal Activity</p>
            <LoadingState />
          </div>

          {/* Pipeline stage progress */}
          <AnalysisPipeline activeStage={activeStage} completed={completed} />

          {/* Error handling UI */}
          {errorMsg && (
            <div className="mt-12 border border-red-500/40 bg-red-500/10 p-6">
              <p className="font-mono text-sm text-red-400 font-semibold mb-2">
                Pipeline Execution Notice
              </p>
              <p className="font-mono text-xs text-muted-foreground mb-4">
                {errorMsg}
              </p>
              <div className="flex gap-4">
                <button
                  type="button"
                  onClick={() => navigate({ to: "/analyze" })}
                  className="px-4 py-2 border border-border text-xs font-mono hover:bg-foreground hover:text-background transition-colors"
                >
                  Return to Upload
                </button>
                <button
                  type="button"
                  onClick={() => navigate({ to: "/results" })}
                  className="px-4 py-2 border border-border text-xs font-mono hover:bg-foreground hover:text-background transition-colors"
                >
                  View Demo Fallback
                </button>
              </div>
            </div>
          )}

          {/* Input file reference */}
          {(file || sampleId) && (
            <div className="mt-16 border-t border-border pt-6">
              <p className="label-mono">Target Input</p>
              <p className="mt-2 truncate font-mono text-[0.8rem] text-muted-foreground">
                {file ? file.name : `Dataset Sample: ${sampleId}`}
              </p>
            </div>
          )}
        </main>
      </div>
    </ThemeProvider>
  );
}
