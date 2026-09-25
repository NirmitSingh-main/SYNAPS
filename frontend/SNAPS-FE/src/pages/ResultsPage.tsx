import { useEffect, useRef } from "react";
import { Link } from "@tanstack/react-router";
import { ThemeProvider } from "@/lib/theme";
import { Header, Footer } from "@/components/si/Chrome";
import { ClassificationCard } from "@/components/ClassificationCard";
import { SignalMetrics } from "@/components/SignalMetrics";
import { SignalWaveform } from "@/components/SignalWaveform";
import { FrequencySpectrum } from "@/components/FrequencySpectrum";
import { Spectrogram } from "@/components/Spectrogram";
import { FeatureTable } from "@/components/FeatureTable";
import { PredictionChart } from "@/components/PredictionChart";
import { ExplanationPanel } from "@/components/ExplanationPanel";
import { PrintReport } from "@/components/report/PrintReport";
import { demoAnalysis } from "@/data/demoData";
import {
  getAnalysisData,
  getAnalysisFile,
  getAnalysisSample,
} from "@/lib/analysisStore";
import { useAuth } from "@/lib/AuthContext";
import { supabase } from "@/lib/supabase";

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
  const { user } = useAuth();
  const liveData = getAnalysisData();
  const file = getAnalysisFile();
  const sampleId = getAnalysisSample();

  const data = liveData || demoAnalysis;
  const isLive = !!liveData;

  // Prevent duplicate saves (React StrictMode guard)
  const savedRef = useRef<unknown>(null);

  useEffect(() => {
    if (!isLive || !user || !data) return;
    if (savedRef.current === data) return;
    savedRef.current = data;

    const currentUser = user;

    async function saveReport() {
      try {
        const { error } = await supabase.from("analysis_reports").insert({
          user_id: currentUser.id,
          filename: file?.name || data.filename || sampleId || "unknown",
          format: data.format || "IQ",
          classification: data.classification,
          confidence: data.confidence,
          sample_rate: data.sampleRate,
          duration: data.duration,
          bandwidth: data.bandwidth,
          snr: data.snr,
          peak_frequency: data.peakFrequency,
          num_samples: data.numSamples,
          prediction_breakdown: data.predictionBreakdown,
          features: data.features,
          explanation: data.explanation,
          raw_report: {
            classification: data.classification,
            confidence: data.confidence,
            sampleRate: data.sampleRate,
            duration: data.duration,
            bandwidth: data.bandwidth,
            snr: data.snr,
            peakFrequency: data.peakFrequency,
            numSamples: data.numSamples,
            predictionBreakdown: data.predictionBreakdown,
            features: data.features,
            explanation: data.explanation,
          },
        });
        if (error) {
          console.error("Failed to save analysis report:", error);
        }
      } catch (err) {
        console.error("Failed to save analysis report:", err);
      }
    }

    saveReport();
  }, [isLive, user, data, file, sampleId]);

  const metrics = [
    { label: "Sample rate", value: formatHz(data.sampleRate), unit: "" },
    { label: "Duration", value: formatDuration(data.duration), unit: "" },
    {
      label: "Occupied Bandwidth (99% Power)",
      value: formatHz(data.bandwidth),
      unit: "",
    },
    {
      label: "SNR",
      value: `${typeof data.snr === "number" ? data.snr.toFixed(1) : data.snr} dB`,
      unit: "",
    },
    {
      label: "Peak frequency",
      value: formatHz(data.peakFrequency),
      unit: "",
    },
    {
      label: "Samples",
      value: formatCount(data.numSamples),
      unit: "",
    },
    ...(typeof data.recoveredBitCount === "number" && data.recoveredBitCount > 0
      ? [
        {
          label: "Recovered Bits",
          value: `${formatCount(data.recoveredBitCount)} bits`,
          unit: "",
        },
        {
          label: "Encoding",
          value: data.dataEncoding || "ASCII",
          unit: "",
        },
        {
          label: "Data Conversion",
          value:
            data.dataConversionValid && data.convertedData
              ? String(data.convertedData)
              : "Conversion Done",
          unit: "",
        },
      ]
      : data.recoveredBitCount === null || data.recoveredBitCount === undefined
        ? [
          {
            label: "Recovered Bits",
            value: "—",
            unit: "",
          },
        ]
        : []),
  ];

  return (
    <ThemeProvider>
      {/* Screen Interactive UI */}
      <div className="screen-only-view relative min-h-screen">
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
                  className={`inline-block rounded-full border px-4 py-2 font-mono text-[0.65rem] tracking-[0.14em] ${isLive
                    ? "border-green-500/40 text-green-400 bg-green-500/10"
                    : "border-dashed border-border-strong text-muted-foreground"
                    }`}
                >
                  {isLive
                    ? "LIVE ENGINE ANALYSIS · CONFIRMED"
                    : "DEMO ANALYSIS · ILLUSTRATIVE DATA"}
                </span>
              </div>
            </div>

            {/* Source file reference */}
            <div className="mt-6 flex items-center gap-3">
              <span className="label-mono">Input Source</span>
              <span className="max-w-md truncate font-mono text-[0.78rem] text-muted-foreground">
                {file
                  ? file.name
                  : sampleId
                    ? `Dataset sample: ${sampleId}`
                    : data.filename}{" "}
                ({data.format || "IQ"})
              </span>
            </div>
          </div>

          <div className="rule-line mb-12" />

          {/* ---- Top row: Classification + Metrics ---- */}
          <div className="grid gap-6 md:grid-cols-2">
            <ClassificationCard
              classification={data.classification}
              confidence={data.confidence}
              signalType={data.signalType}
              detectedComponents={data.detectedComponents}
              isDemoData={!isLive}
            />
            <SignalMetrics
              metrics={metrics}
              isDemoData={!isLive}
            />
          </div>

          {/* ---- Bit Recovery & Component Analysis ---- */}
          {data.bitRecovery && (
            <div className="mt-6 floating-surface p-6">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-4">
                <p className="label-mono">
                  {data.signalType === "MIXED"
                    ? "Multi-component signal analysis"
                    : "Bit recovery & ground-truth validation"}
                </p>
                <span
                  className={`font-mono text-[0.68rem] px-2.5 py-1 border ${data.bitRecovery.validation_status === "VALIDATED"
                    ? "border-green-500/40 text-green-400 bg-green-500/10"
                    : (data.bitRecovery.validation_status === "Component bit recovery not validated"
                      || data.bitRecovery.validation_status === "COMPONENT_RECOVERY_NOT_VALIDATED")
                      ? "border-signal/40 text-signal bg-signal/10"
                      : "border-border text-muted-foreground"
                    }`}
                >
                  {data.bitRecovery.validation_status}
                </span>
              </div>

              {data.signalType === "MIXED" ? (
                <div className="mt-4 space-y-4">
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Composite multi-carrier signal detected. The waveform is analyzed and preserved intact. Component bit retrieval is not claimed without dedicated separation filtering.
                  </p>
                  {data.componentResults && data.componentResults.length > 0 && (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left font-mono text-xs">
                        <thead>
                          <tr className="border-b border-border text-muted-foreground">
                            <th className="py-2 pr-4">#</th>
                            <th className="py-2 pr-4">Component</th>
                            <th className="py-2 pr-4">Symbol Rate</th>
                            <th className="py-2 pr-4">Freq Offset</th>
                            <th className="py-2 pr-4">Placement</th>
                            <th className="py-2">Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {data.componentResults.map((comp) => (
                            <tr key={comp.componentIndex} className="border-b border-border/50">
                              <td className="py-2 pr-4 text-signal">0{comp.componentIndex}</td>
                              <td className="py-2 pr-4 font-semibold text-foreground">{comp.modulation}</td>
                              <td className="py-2 pr-4 text-muted-foreground">{comp.symbolRate ? `${(comp.symbolRate / 1e3).toFixed(0)} kBaud` : "—"}</td>
                              <td className="py-2 pr-4 text-muted-foreground">{comp.frequencyOffset ? `${(comp.frequencyOffset / 1e3).toFixed(1)} kHz` : "0.0 kHz"}</td>
                              <td className="py-2 pr-4 text-muted-foreground">{comp.frequencyPlacement ? `${(comp.frequencyPlacement / 1e3).toFixed(1)} kHz` : "0.0 kHz"}</td>
                              <td className="py-2 text-muted-foreground text-[0.7rem]">{comp.status}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              ) : (
                <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-4 font-mono text-xs">
                  <div>
                    <span className="text-muted-foreground block text-[0.65rem] tracking-wider uppercase">Reference Bits</span>
                    <span className="text-foreground text-sm font-medium">{data.bitRecovery.reference_bit_count ?? "—"}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block text-[0.65rem] tracking-wider uppercase">Recovered Bits</span>
                    <span className="text-foreground text-sm font-medium">{data.bitRecovery.recovered_bit_count ?? "—"}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block text-[0.65rem] tracking-wider uppercase">Bit Accuracy</span>
                    <span className="text-foreground text-sm font-medium">
                      {data.bitRecovery.bit_accuracy_pct !== undefined && data.bitRecovery.bit_accuracy_pct !== null
                        ? `${data.bitRecovery.bit_accuracy_pct.toFixed(1)}%`
                        : "—"}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block text-[0.65rem] tracking-wider uppercase">Bit Error Rate (BER)</span>
                    <span className="text-foreground text-sm font-medium">
                      {data.bitRecovery.ber !== undefined && data.bitRecovery.ber !== null
                        ? data.bitRecovery.ber.toFixed(4)
                        : "—"}
                    </span>
                  </div>
                  {data.convertedData && (
                    <div className="col-span-2 sm:col-span-4 mt-2 pt-3 border-t border-border/60">
                      <span className="text-muted-foreground block text-[0.65rem] tracking-wider uppercase mb-1">
                        Decoded Message Payload ({data.dataEncoding || "ASCII"})
                      </span>
                      <pre className="p-2.5 bg-muted/30 border border-border/80 font-mono text-xs text-foreground overflow-x-auto whitespace-pre-wrap">
                        {data.convertedData}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* ---- Waveform ---- */}
          <div className="mt-6">
            <SignalWaveform
              samples={data.waveformSamples}
              {...(data.waveformTime ? { time: data.waveformTime } : {})}
              isDemoData={!isLive}
            />
          </div>

          {/* ---- Spectrum + Spectrogram ---- */}
          <div className="mt-6 grid gap-6 md:grid-cols-2">
            <FrequencySpectrum
              bins={data.spectrumBins}
              {...(data.spectrumFrequencies ? { frequencies: data.spectrumFrequencies } : {})}
              isDemoData={!isLive}
            />

            <Spectrogram
              rows={data.spectrogramRows}
              {...(data.spectrogramTimes ? { times: data.spectrogramTimes } : {})}
              {...(data.spectrogramFrequencies ? { frequencies: data.spectrogramFrequencies } : {})}
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
              className="border border-border rounded-lg px-4 py-2 font-mono text-[0.75rem] text-muted-foreground transition-colors duration-200 hover:border-foreground hover:text-foreground"
            >
              Export report
            </button>
          </div>
        </main>
        <Footer />
      </div>

      {/* Dedicated Print Report UI (shown only when printing / saving as PDF) */}
      <PrintReport
        data={data}
        file={file}
        sampleId={sampleId}
        isLive={isLive}
      />
    </ThemeProvider>
  );
}