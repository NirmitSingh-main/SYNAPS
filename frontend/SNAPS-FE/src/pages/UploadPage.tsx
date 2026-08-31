import { useState, useEffect } from "react";
import { useNavigate } from "@tanstack/react-router";
import { ThemeProvider } from "@/lib/theme";
import { Header } from "@/components/si/Chrome";
import { FileUploader } from "@/components/FileUploader";
import { FileInfoCard } from "@/components/FileInfoCard";
import { setAnalysisFile, setAnalysisSample } from "@/lib/analysisStore";
import { checkBackendHealth, getSampleSignals, SampleSignalItem } from "@/services/api";

export function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [selectedSample, setSelectedSample] = useState<SampleSignalItem | null>(null);
  const [samples, setSamples] = useState<SampleSignalItem[]>([]);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [activeTab, setActiveTab] = useState<"upload" | "samples">("upload");
  const navigate = useNavigate();

  useEffect(() => {
    checkBackendHealth().then((h) => {
      setBackendOnline(h.status === "ONLINE");
    });
    getSampleSignals().then((list) => {
      if (list.length > 0) setSamples(list);
    });
  }, []);

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

  return (
    <ThemeProvider>
      <div className="relative min-h-screen">
        <Header />

        <main className="mx-auto max-w-3xl px-6 pb-32 pt-32">
          {/* Page heading & Backend status */}
          <div className="mb-12 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <p className="label-mono">Signal Intelligence</p>
              <h1 className="mt-4 text-3xl tracking-[-0.02em] sm:text-4xl">
                Upload or select a signal to analyze.
              </h1>
            </div>

            {/* Backend connection pill */}
            <div className="shrink-0 pt-2">
              <span
                className={`inline-flex items-center gap-2 border px-3 py-1.5 font-mono text-[0.7rem] tracking-[0.1em] ${
                  backendOnline === true
                    ? "border-green-500/40 text-green-400 bg-green-500/5"
                    : backendOnline === false
                    ? "border-amber-500/40 text-amber-400 bg-amber-500/5"
                    : "border-border text-muted-foreground"
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    backendOnline === true
                      ? "bg-green-400 animate-pulse"
                      : backendOnline === false
                      ? "bg-amber-400"
                      : "bg-muted-foreground"
                  }`}
                />
                {backendOnline === true
                  ? "AI DSP BACKEND ONLINE"
                  : backendOnline === false
                  ? "STANDALONE / OFFLINE"
                  : "CONNECTING..."}
              </span>
            </div>
          </div>

          <div className="rule-line mb-8" />

          {/* Mode Switcher Tabs */}
          <div className="mb-8 flex gap-4 border-b border-border pb-4">
            <button
              type="button"
              onClick={() => {
                setActiveTab("upload");
                setSelectedSample(null);
              }}
              className={`font-mono text-xs tracking-wider uppercase transition-colors ${
                activeTab === "upload"
                  ? "text-foreground font-semibold border-b-2 border-signal pb-1"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Custom File Upload
            </button>
            <button
              type="button"
              onClick={() => {
                setActiveTab("samples");
                setFile(null);
              }}
              className={`font-mono text-xs tracking-wider uppercase transition-colors ${
                activeTab === "samples"
                  ? "text-foreground font-semibold border-b-2 border-signal pb-1"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Preloaded Dataset Samples ({samples.length})
            </button>
          </div>

          {/* Tab 1: File Upload */}
          {activeTab === "upload" && (
            <div>
              {file ? (
                <div className="space-y-6">
                  <FileInfoCard file={file} onRemove={handleRemove} />
                </div>
              ) : (
                <FileUploader onFileSelected={handleFileSelected} />
              )}
            </div>
          )}

          {/* Tab 2: Preloaded Dataset Samples */}
          {activeTab === "samples" && (
            <div className="space-y-4">
              <p className="text-xs text-muted-foreground font-mono">
                Select an IQ / WAV dataset sample to run full AI Transformer & DSP analysis:
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
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
                      className={`p-4 border transition-all cursor-pointer ${
                        isSelected
                          ? "border-signal bg-signal/10"
                          : "border-border hover:border-border-strong bg-background"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-xs font-semibold text-foreground">
                          {s.modulation}
                        </span>
                        <span className="font-mono text-[0.65rem] border border-border px-1.5 py-0.5 text-muted-foreground uppercase">
                          {s.format}
                        </span>
                      </div>
                      <p className="mt-2 font-mono text-[0.75rem] text-muted-foreground truncate">
                        {s.filename}
                      </p>
                      <p className="mt-1 font-mono text-[0.65rem] text-muted-foreground">
                        {(s.size_bytes / 1024).toFixed(1)} KB · 8,192 complex samples
                      </p>
                    </div>
                  );
                })}
              </div>

              {samples.length === 0 && (
                <div className="border border-dashed border-border p-6 text-center text-xs font-mono text-muted-foreground">
                  Loading dataset samples from backend...
                </div>
              )}
            </div>
          )}

          {/* Start analysis CTA */}
          <div className="mt-10 flex items-center justify-between">
            <div className="text-xs font-mono text-muted-foreground">
              {file && <span>Selected: {file.name}</span>}
              {selectedSample && <span>Selected: {selectedSample.filename} ({selectedSample.modulation})</span>}
            </div>

            <button
              type="button"
              onClick={handleStartAnalysis}
              disabled={!isReady}
              aria-disabled={!isReady}
              className="hover-arrow inline-flex items-center justify-between gap-8 border border-border-strong px-6 py-4 text-[0.95rem] transition-colors duration-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-35"
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
              <span>Run Master Intelligence Pipeline</span>
              <span className="arrow font-mono" style={{ color: isReady ? "var(--background)" : "var(--signal)" }}>
                →
              </span>
            </button>
          </div>

          {/* Accepted formats note */}
          {!file && !selectedSample && (
            <p className="mt-6 text-right font-mono text-[0.68rem] tracking-[0.1em] text-muted-foreground">
              Accepts Raw IQ (float32 interleaved) · WAV stereo
            </p>
          )}
        </main>
      </div>
    </ThemeProvider>
  );
}
