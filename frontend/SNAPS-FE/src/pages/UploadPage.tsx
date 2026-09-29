import { useState, useEffect } from "react";
import { useNavigate } from "@tanstack/react-router";
import { ThemeProvider } from "@/lib/theme";
import { Header, Footer } from "@/components/si/Chrome";
import { FileUploader } from "@/components/FileUploader";
import { FileInfoCard } from "@/components/FileInfoCard";
import { setAnalysisFile, setAnalysisSample } from "@/lib/analysisStore";
import { checkBackendHealth, getSampleSignals, SampleSignalItem } from "@/services/api";
import { useAuth } from "@/lib/AuthContext";

const MODULATION_CLASSES = ["BPSK", "QPSK", "FSK", "QAM16"];

export function UploadPage() {
  const { user, loading } = useAuth();
  const [file, setFile] = useState<File | null>(null);
  const [selectedSample, setSelectedSample] = useState<SampleSignalItem | null>(null);
  const [samples, setSamples] = useState<SampleSignalItem[]>([]);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [activeTab, setActiveTab] = useState<"upload" | "samples">("upload");
  const navigate = useNavigate();

  // Auth guard
  useEffect(() => {
    if (!loading && !user) {
      navigate({ to: "/auth" });
    }
  }, [user, loading, navigate]);

  useEffect(() => {
    if (!user) return;
    checkBackendHealth().then((h) => {
      setBackendOnline(h.status === "ONLINE");
    });
    getSampleSignals().then((list) => {
      if (list.length > 0) setSamples(list);
    });
  }, [user]);

  const handleFileSelected = (f: File) => {
    setFile(f);
    setSelectedSample(null);
  };

  const handleRemove = () => {
    setFile(null);
    setSelectedSample(null);
  };

  const handleSelectSample = (sample: SampleSignalItem) => {
    setSelectedSample(sample);
    setFile(null);
  };

  const handleStartAnalysis = () => {
    if (file) {
      setAnalysisFile(file);
      navigate({ to: "/processing" });
    } else if (selectedSample) {
      setAnalysisSample(selectedSample.sample_id);
      navigate({ to: "/processing" });
    }
  };

  const isReady = !!(file || selectedSample);

  // While auth is loading or not logged in, render nothing (redirect will happen)
  if (loading || !user) return null;

  return (
    <ThemeProvider>
      <div className="relative min-h-screen overflow-hidden">
        {/* Subtle technical grid background */}
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.025]"
          style={{
            backgroundImage: `repeating-linear-gradient(0deg, transparent, transparent 39px, var(--color-border-strong) 39px, var(--color-border-strong) 40px), repeating-linear-gradient(90deg, transparent, transparent 39px, var(--color-border-strong) 39px, var(--color-border-strong) 40px)`,
          }}
        />

        <Header />

        <main className="relative mx-auto max-w-2xl px-6 pb-32 pt-28">
          {/* Page heading */}
          <div className="mb-8 flex items-start justify-between">
            <div>
              <p className="label-mono">Signal Intelligence</p>
              <h1 className="mt-3 text-2xl tracking-[-0.02em] sm:text-3xl">
                Analyze
              </h1>
            </div>

            {/* Engine status — compact */}
            <div className="mt-1 shrink-0">
              <span
                className={`inline-flex items-center gap-1.5 font-mono text-[0.65rem] tracking-[0.12em] ${backendOnline === true
                    ? "text-green-400"
                    : backendOnline === false
                      ? "text-amber-400"
                      : "text-muted-foreground"
                  }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${backendOnline === true
                      ? "bg-green-400 animate-pulse"
                      : backendOnline === false
                        ? "bg-amber-400"
                        : "bg-muted-foreground/50"
                    }`}
                />
                {backendOnline === true
                  ? "ENGINE READY"
                  : backendOnline === false
                    ? "ENGINE OFFLINE"
                    : "CONNECTING"}
              </span>
            </div>
          </div>

          <div className="rule-line mb-8" />

          {/* Pipeline status strip */}
          <div className="mb-8 flex items-center gap-0 overflow-x-auto">
            {["INPUT", "DSP", "FEATURES", "AI"].map((stage, i) => (
              <div
                key={stage}
                className={`flex items-center ${i < 3 ? "flex-1" : ""}`}
              >
                <div className="flex flex-col items-center gap-1">
                  <span className="label-mono text-[0.58rem]">{stage}</span>
                  <span
                    className={`font-mono text-[0.6rem] tracking-widest ${isReady && stage === "INPUT"
                        ? "text-green-400"
                        : backendOnline === true
                          ? "text-signal"
                          : "text-muted-foreground/40"
                      }`}
                  >
                    {isReady && stage === "INPUT" ? "LOADED" : "READY"}
                  </span>
                </div>
                {i < 3 && (
                  <div className="mx-3 h-px flex-1 bg-border" />
                )}
              </div>
            ))}
          </div>

          {/* Mode tabs */}
          <div className="mb-6 flex gap-0 border-b border-border">
            <button
              type="button"
              onClick={() => { setActiveTab("upload"); setSelectedSample(null); }}
              className={`pb-3 pr-6 font-mono text-[0.72rem] tracking-widest uppercase transition-colors ${activeTab === "upload"
                  ? "border-b-2 border-signal text-foreground"
                  : "text-muted-foreground hover:text-foreground"
                }`}
            >
              Upload
            </button>
            <button
              type="button"
              onClick={() => { setActiveTab("samples"); setFile(null); }}
              className={`pb-3 pr-6 font-mono text-[0.72rem] tracking-widest uppercase transition-colors ${activeTab === "samples"
                  ? "border-b-2 border-signal text-foreground"
                  : "text-muted-foreground hover:text-foreground"
                }`}
            >
              Dataset ({samples.length})
            </button>
          </div>

          {/* Tab: Upload — floating dropzone */}
          {activeTab === "upload" && (
            <div>
              {file ? (
                <FileInfoCard file={file} onRemove={handleRemove} />
              ) : (
                <div className="relative floating-surface p-2">
                  <FileUploader onFileSelected={handleFileSelected} />
                </div>
              )}
            </div>
          )}

          {/* Tab: Dataset samples */}
          {activeTab === "samples" && (
            <div className="space-y-3">
              {/* Compact modulation class pills */}
              <div className="mb-4 flex gap-2">
                {MODULATION_CLASSES.map((mod) => {
                  const modSamples = samples.filter((s) =>
                    s.modulation.toUpperCase().includes(mod)
                  );
                  return (
                    <span
                      key={mod}
                      className="border border-border px-2.5 py-1 font-mono text-[0.68rem] tracking-wider text-muted-foreground"
                    >
                      {mod}
                      {modSamples.length > 0 && (
                        <span className="ml-1.5 text-signal">{modSamples.length}</span>
                      )}
                    </span>
                  );
                })}
              </div>

              {/* Sample grid */}
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {samples.map((s) => {
                  const isSelected = selectedSample?.sample_id === s.sample_id;
                  return (
                    <div
                      key={s.sample_id}
                      role="button"
                      tabIndex={0}
                      onClick={() => handleSelectSample(s)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") handleSelectSample(s);
                      }}
                      className={`cursor-pointer floating-surface floating-surface-hover p-3.5 transition-all ${isSelected
                          ? "!border-signal bg-signal/10"
                          : ""
                        }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-[0.75rem] font-medium text-foreground">
                          {s.modulation}
                        </span>
                        <span className="font-mono text-[0.62rem] text-muted-foreground uppercase">
                          {s.format}
                        </span>
                      </div>
                      <p className="mt-1 truncate font-mono text-[0.68rem] text-muted-foreground">
                        {s.filename}
                      </p>
                      <p className="mt-0.5 font-mono text-[0.62rem] text-muted-foreground/60">
                        {(s.size_bytes / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  );
                })}
              </div>

              {samples.length === 0 && (
                <div className="border border-dashed border-border p-8 text-center">
                  <p className="label-mono">Backend offline — dataset unavailable.</p>
                </div>
              )}
            </div>
          )}

          {/* CTA */}
          <div className="mt-8 flex items-center justify-between">
            <div className="min-w-0 font-mono text-[0.72rem] text-muted-foreground">
              {file && <span className="truncate">→ {file.name}</span>}
              {selectedSample && (
                <span className="truncate">
                  → {selectedSample.filename}
                </span>
              )}
            </div>

            <button
              type="button"
              onClick={handleStartAnalysis}
              disabled={!isReady}
              aria-disabled={!isReady}
              className="hover-arrow inline-flex shrink-0 items-center justify-between gap-8 border border-border-strong px-6 py-4 font-mono text-[0.85rem] transition-colors duration-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-30"
              style={
                isReady
                  ? {
                    backgroundColor: "var(--foreground)",
                    color: "var(--background)",
                    borderColor: "var(--foreground)",
                  }
                  : {}
              }
            >
              <span>Run Analysis</span>
              <span
                className="arrow font-mono"
                style={{ color: isReady ? "var(--background)" : "var(--signal)" }}
              >
                →
              </span>
            </button>
          </div>

          {!file && !selectedSample && (
            <p className="mt-4 text-right font-mono text-[0.65rem] tracking-[0.1em] text-muted-foreground/60">
              IQ · WAV
            </p>
          )}
        </main>
        <Footer />
      </div>
    </ThemeProvider>
  );
}
