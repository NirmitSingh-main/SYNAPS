import { Link } from "@tanstack/react-router";
import { ThemeProvider } from "@/lib/theme";
import { Header } from "@/components/si/Chrome";
import { ClassificationCard } from "@/components/ClassificationCard";
import { SignalMetrics } from "@/components/SignalMetrics";
import { SignalWaveform } from "@/components/SignalWaveform";
import { FrequencySpectrum } from "@/components/FrequencySpectrum";
import { Spectrogram } from "@/components/Spectrogram";
import { FeatureTable } from "@/components/FeatureTable";
import { PredictionChart } from "@/components/PredictionChart";
import { ExplanationPanel } from "@/components/ExplanationPanel";
import { demoAnalysis } from "@/data/demoData";
import { getAnalysisData, getAnalysisFile, getAnalysisSample } from "@/lib/analysisStore";

function formatHz(hz: number): string {
  if (!hz || isNaN(hz)) return "0 Hz";
  if (hz >= 1_000_000) return `${(hz / 1_000_000).toFixed(2)} MHz`;
  if (hz >= 1_000) return `${(hz / 1_000).toFixed(1)} kHz`;
  return `${hz.toFixed(0)} Hz`;
}

function formatDuration(s: number): string {
  if (!s || isNaN(s)) return "0 ms";
  return `${(s * 1000).toFixed(1)} ms`;
}

function formatCount(n: number): string {
  if (!n || isNaN(n)) return "0";
  return n.toLocaleString("en-US");
}

export function ResultsPage() {
  const liveData = getAnalysisData();
  const file = getAnalysisFile();
  const sampleId = getAnalysisSample();

  // Use live backend data if available, otherwise fallback to demoAnalysis
  const data = liveData || demoAnalysis;
  const isLive = !!liveData;

  const metrics = [
    { label: "Sample rate",    value: formatHz(data.sampleRate),      unit: "" },
    { label: "Duration",       value: formatDuration(data.duration),   unit: "" },
    { label: "Bandwidth (3dB)",value: formatHz(data.bandwidth),        unit: "" },
    { label: "SNR",            value: `${typeof data.snr === "number" ? data.snr.toFixed(1) : data.snr} dB`, unit: "" },
    { label: "Peak frequency", value: formatHz(data.peakFrequency),    unit: "" },
    { label: "Samples",        value: formatCount(data.numSamples),     unit: "" },
  ];

  return (
    <ThemeProvider>
      <div className="relative min-h-screen">
        <Header />

        <main className="mx-auto max-w-4xl px-6 pb-32 pt-32">
          {/* ---- Page header ---- */}
          <div className="mb-12">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <p className="label-mono">Analysis result</p>
                <h1 className="mt-4 text-3xl tracking-[-0.02em] sm:text-4xl">
                  Signal intelligence report
                </h1>
              </div>

              {/* Status badge */}
              <div className="shrink-0">
                <span
                  className={`inline-block border px-4 py-2 font-mono text-[0.65rem] tracking-[0.14em] ${
                    isLive
                      ? "border-green-500/40 text-green-400 bg-green-500/10"
                      : "border-dashed border-border-strong text-muted-foreground"
                  }`}
                >
                  {isLive ? "LIVE ENGINE ANALYSIS · CONFIRMED" : "DEMO ANALYSIS · ILLUSTRATIVE DATA"}
                </span>
              </div>
            </div>

            {/* Source file reference */}
            <div className="mt-6 flex items-center gap-3">
              <span className="label-mono">Input Source</span>
              <span className="font-mono text-[0.78rem] text-muted-foreground truncate max-w-md">
                {file ? file.name : sampleId ? `Dataset sample: ${sampleId}` : data.filename} ({data.format || "IQ"})
              </span>
            </div>
          </div>

          <div className="rule-line mb-12" />

          {/* ---- Top row: Classification + Metrics ---- */}
          <div className="grid gap-6 md:grid-cols-2">
            <ClassificationCard
              classification={data.classification}
              confidence={data.confidence}
              isDemoData={!isLive}
            />
            <SignalMetrics
              metrics={metrics}
              isDemoData={!isLive}
            />
          </div>

          {/* ---- Waveform ---- */}
          <div className="mt-6">
            <SignalWaveform
              samples={data.waveformSamples}
              isDemoData={!isLive}
            />
          </div>

          {/* ---- Spectrum + Spectrogram ---- */}
          <div className="mt-6 grid gap-6 md:grid-cols-2">
            <FrequencySpectrum
              bins={data.spectrumBins}
              isDemoData={!isLive}
            />
            <Spectrogram
              rows={data.spectrogramRows}
              isDemoData={!isLive}
            />
          </div>

          {/* ---- Prediction breakdown ---- */}
          <div className="mt-6">
            <PredictionChart
              predictions={data.predictionBreakdown}
              isDemoData={!isLive}
            />
          </div>

          {/* ---- Feature table ---- */}
          <div className="mt-6">
            <FeatureTable
              features={data.features}
              isDemoData={!isLive}
            />
          </div>

          {/* ---- Intelligence explanation ---- */}
          <div className="mt-6">
            <ExplanationPanel
              text={data.explanation}
              isDemoData={!isLive}
            />
          </div>

          {/* ---- Actions footer ---- */}
          <div className="mt-16 flex items-center justify-between border-t border-border pt-8">
            <Link
              to="/analyze"
              className="hover-arrow inline-flex items-center gap-3 font-mono text-[0.8rem] text-muted-foreground transition-colors duration-200 hover:text-foreground"
            >
              <span className="arrow">←</span>
              <span>Analyze another signal</span>
            </Link>

            <button
              type="button"
              onClick={() => window.print()}
              className="border border-border px-4 py-2 font-mono text-[0.75rem] text-muted-foreground transition-colors duration-200 hover:border-foreground hover:text-foreground"
            >
              Export report
            </button>
          </div>
        </main>
      </div>
    </ThemeProvider>
  );
}
