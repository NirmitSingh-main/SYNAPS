import React, { useEffect, useRef } from "react";

interface Feature {
  name: string;
  value: string;
  unit: string;
  description: string;
}

interface Prediction {
  classLabel: string;
  probability: number;
}

interface ComponentResult {
  componentIndex: number;
  modulation: string;
  symbolRate?: number | null | undefined;
  frequencyOffset?: number | null | undefined;
  frequencyPlacement?: number | null | undefined;
  status: string;
}

interface BitRecoveryData {
  validation_status: string;
  reference_bit_count?: number | null | undefined;
  recovered_bit_count?: number | null | undefined;
  bit_accuracy_pct?: number | null | undefined;
  ber?: number | null | undefined;
}

export interface PrintReportData {
  filename?: string | null | undefined;
  format?: string | null | undefined;
  classification: string;
  confidence: number;
  sampleRate: number;
  duration: number;
  bandwidth: number;
  snr: number;
  peakFrequency: number;
  numSamples: number;
  signalType?: string | null | undefined;
  detectedComponents?: string[] | null | undefined;
  predictionBreakdown?: Prediction[] | null | undefined;
  features?: Feature[] | null | undefined;
  explanation?: string | null | undefined;
  waveformSamples?: number[] | null | undefined;
  waveformTime?: number[] | null | undefined;
  spectrumBins?: number[] | null | undefined;
  spectrumFrequencies?: number[] | null | undefined;
  spectrogramRows?: number[][] | null | undefined;
  spectrogramTimes?: number[] | null | undefined;
  spectrogramFrequencies?: number[] | null | undefined;
  bitRecovery?: BitRecoveryData | null | undefined;
  recoveredBitCount?: number | null | undefined;
  dataEncoding?: string | null | undefined;
  dataConversionValid?: boolean | null | undefined;
  convertedData?: string | null | undefined;
  componentResults?: ComponentResult[] | null | undefined;
}

interface PrintReportProps {
  data: PrintReportData;
  file?: File | null;
  sampleId?: string | null;
  isLive: boolean;
}

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

function formatFreq(hz: number): string {
  if (Math.abs(hz) >= 1_000_000) return `${(hz / 1_000_000).toFixed(2)} MHz`;
  if (Math.abs(hz) >= 1_000) return `${(hz / 1_000).toFixed(1)} kHz`;
  return `${hz.toFixed(0)} Hz`;
}

/* -------------------------------------------------------------------------- */
/* PRINT WAVEFORM CANVAS                                                      */
/* -------------------------------------------------------------------------- */
function PrintWaveformPlot({
  samples,
  time,
}: {
  samples?: number[] | null | undefined;
  time?: number[] | null | undefined;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !samples || samples.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);

    // Subtle technical grid
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    for (let x = w / 4; x < w; x += w / 4) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    ctx.stroke();

    // High-contrast clean trace
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1.25;
    ctx.beginPath();

    const step = samples.length / w;
    for (let i = 0; i < w; i++) {
      const idx = Math.floor(i * step);
      const val = samples[idx] ?? 0;
      const y = h / 2 - (val * (h * 0.42));
      if (i === 0) ctx.moveTo(i, y);
      else ctx.lineTo(i, y);
    }
    ctx.stroke();
  }, [samples]);

  const start = time?.[0] ?? 0;
  const end = time?.[time.length - 1] ?? 0;

  return (
    <div>
      <div className="border border-slate-300 rounded p-1.5 bg-white">
        <canvas ref={canvasRef} width={760} height={110} className="w-full h-24 block" />
      </div>
      <div className="mt-1 flex justify-between text-[0.68rem] text-slate-500 font-mono">
        <span>{time ? `t = ${(start * 1000).toFixed(1)} ms` : "t = 0.0 ms"}</span>
        <span>Normalized Amplitude</span>
        <span>{time ? `t = ${(end * 1000).toFixed(1)} ms` : "t = N ms"}</span>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* PRINT SPECTRUM CANVAS                                                      */
/* -------------------------------------------------------------------------- */
function PrintSpectrumPlot({
  bins,
  frequencies,
}: {
  bins?: number[] | null | undefined;
  frequencies?: number[] | null | undefined;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !bins || bins.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h - 1);
    ctx.lineTo(w, h - 1);
    for (let x = w / 4; x < w; x += w / 4) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    ctx.stroke();

    // Subtle area fill
    const step = bins.length / w;
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < w; i++) {
      const idx = Math.floor(i * step);
      const val = Math.max(0, Math.min(1, bins[idx] ?? 0));
      const y = h - (val * (h - 4));
      ctx.lineTo(i, y);
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fillStyle = "rgba(30, 41, 59, 0.06)";
    ctx.fill();

    // Spectrum trace
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    for (let i = 0; i < w; i++) {
      const idx = Math.floor(i * step);
      const val = Math.max(0, Math.min(1, bins[idx] ?? 0));
      const y = h - (val * (h - 4));
      if (i === 0) ctx.moveTo(i, y);
      else ctx.lineTo(i, y);
    }
    ctx.stroke();
  }, [bins]);

  const first = frequencies?.[0] ?? 0;
  const last = frequencies?.[frequencies.length - 1] ?? 0;

  return (
    <div>
      <div className="border border-slate-300 rounded p-1.5 bg-white">
        <canvas ref={canvasRef} width={360} height={100} className="w-full h-22 block" />
      </div>
      <div className="mt-1 flex justify-between text-[0.68rem] text-slate-500 font-mono">
        <span>{formatFreq(first)}</span>
        <span>Normalized Power</span>
        <span>{formatFreq(last)}</span>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* PRINT SPECTROGRAM CANVAS                                                   */
/* -------------------------------------------------------------------------- */
function PrintSpectrogramPlot({
  rows,
  times,
  frequencies,
}: {
  rows?: number[][] | null | undefined;
  times?: number[] | null | undefined;
  frequencies?: number[] | null | undefined;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !rows || rows.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    const numRows = rows.length;
    const numCols = rows[0]?.length || 1;
    const cellW = w / numCols;
    const cellH = h / numRows;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);

    for (let r = 0; r < numRows; r++) {
      const row = rows[r];
      if (!row) continue;
      for (let c = 0; c < numCols; c++) {
        const val = Math.max(0, Math.min(1, row[c] ?? 0));
        // High-contrast clean scientific gradient
        if (val < 0.15) {
          ctx.fillStyle = "#f8fafc";
        } else if (val < 0.45) {
          const intensity = Math.floor((val - 0.15) * 300);
          ctx.fillStyle = `rgb(${220 - intensity}, ${230 - intensity}, 245)`;
        } else if (val < 0.75) {
          const intensity = Math.floor((val - 0.45) * 250);
          ctx.fillStyle = `rgb(${50 + intensity}, ${120 + intensity / 2}, 180)`;
        } else {
          ctx.fillStyle = "#0f172a";
        }
        ctx.fillRect(c * cellW, (numRows - 1 - r) * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }, [rows]);

  const startTime = times?.[0] ?? 0;
  const endTime = times?.[times.length - 1] ?? 0;
  const lowFreq = frequencies?.[0] ?? 0;
  const highFreq = frequencies?.[frequencies.length - 1] ?? 0;

  return (
    <div>
      <div className="border border-slate-300 rounded p-1.5 bg-white">
        <canvas ref={canvasRef} width={360} height={100} className="w-full h-22 block" />
      </div>
      <div className="mt-1 flex justify-between text-[0.68rem] text-slate-500 font-mono">
        <span>{(startTime * 1000).toFixed(0)} ms · {formatFreq(lowFreq)}</span>
        <span>Time × Freq</span>
        <span>{(endTime * 1000).toFixed(0)} ms · {formatFreq(highFreq)}</span>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* MAIN PRINT REPORT COMPONENT — 3-PAGE ENGINEERING DOCUMENT                  */
/* -------------------------------------------------------------------------- */
export function PrintReport({ data, file, sampleId, isLive }: PrintReportProps) {
  const reportDate = new Date().toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });

  const sourceName = file?.name || sampleId || data.filename || "signal_capture.iq";
  const sourceFormat = data.format || "IQ";
  const pct = Math.round(data.confidence * 100);

  // Extract primary features for quick summary
  const featureMap = new Map<string, Feature>();
  data.features?.forEach((f) => featureMap.set(f.name, f));

  const cfo = featureMap.get("Carrier Frequency Offset")?.value || "—";
  const symbolRate = featureMap.get("Symbol Rate")?.value || "—";
  const c40 = featureMap.get("Cumulant C40")?.value || "—";
  const c42 = featureMap.get("Cumulant C42")?.value || "—";
  const papr = featureMap.get("Peak-to-Average Power Ratio")?.value || "—";
  const fingerprint = featureMap.get("Emitter RF Fingerprint ID")?.value || "—";

  return (
    <div className="print-only-report font-sans text-slate-900 bg-white">
      {/* ==================================================================== */}
      {/* PAGE 1: SIGNAL CLASSIFICATION & CORE METRICS                         */}
      {/* ==================================================================== */}
      <div className="print-page flex flex-col justify-between">
        <div>
          {/* Header */}
          <header className="border-b border-slate-900 pb-3 mb-6">
            <div className="flex items-baseline justify-between">
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-slate-950 font-sans">
                  SYNAPS
                </h1>
                <p className="text-xs font-medium text-slate-600 tracking-wide uppercase mt-0.5">
                  Signal Intelligence Report
                </p>
              </div>

              <div className="text-right text-xs text-slate-600 space-y-0.5 font-sans">
                <p><span className="text-slate-400">Date:</span> <span className="font-mono text-slate-800">{reportDate}</span></p>
                <p><span className="text-slate-400">Source:</span> <span className="font-mono text-slate-800">{sourceName}</span> ({sourceFormat})</p>
                <p>
                  <span className="text-slate-400">Status:</span>{" "}
                  <span className="font-medium text-slate-800">
                    {isLive ? "Engine Analysis (Live)" : "Demo Evaluation"}
                  </span>
                </p>
              </div>
            </div>
          </header>

          {/* Section 1: Signal Classification */}
          <section className="mb-8">
            <div className="border-b border-slate-200 pb-1.5 mb-4">
              <h2 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                Signal Classification
              </h2>
            </div>

            <div className="flex items-baseline justify-between mb-4">
              <div>
                <span className="text-4xl font-bold tracking-tight text-slate-950 font-sans">
                  {data.classification}
                </span>
                <p className="text-xs text-slate-500 mt-1 font-sans">
                  {data.signalType === "MIXED"
                    ? "Multi-carrier composite waveform"
                    : "Primary modulation classification"}
                </p>
              </div>

              <div className="text-right">
                <span className="text-3xl font-bold font-mono text-slate-900">
                  {pct}%
                </span>
                <p className="text-xs text-slate-500 mt-1 font-sans">Confidence</p>
              </div>
            </div>

            {/* Probability distribution row */}
            {data.predictionBreakdown && data.predictionBreakdown.length > 0 && (
              <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
                <span className="text-slate-500 font-sans">Class Probability Distribution:</span>
                <div className="flex gap-6 font-mono text-slate-700">
                  {data.predictionBreakdown.map((p) => (
                    <div key={p.classLabel} className="flex items-center gap-1.5">
                      <span className="text-slate-500 font-sans">{p.classLabel}:</span>
                      <span className={p.classLabel === data.classification ? "font-bold text-slate-950" : ""}>
                        {(p.probability * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>

          {/* Section 2: Core Signal Metrics */}
          <section className="mb-8">
            <div className="border-b border-slate-200 pb-1.5 mb-4">
              <h2 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                Core Signal Metrics
              </h2>
            </div>

            <div className="grid grid-cols-2 gap-x-12 gap-y-3.5 text-xs">
              <div className="flex items-baseline justify-between border-b border-slate-100 pb-1">
                <span className="text-slate-600 font-sans">Sample Rate</span>
                <span className="font-mono text-slate-900 font-medium">{formatHz(data.sampleRate)}</span>
              </div>

              <div className="flex items-baseline justify-between border-b border-slate-100 pb-1">
                <span className="text-slate-600 font-sans">Signal Duration</span>
                <span className="font-mono text-slate-900 font-medium">{formatDuration(data.duration)}</span>
              </div>

              <div className="flex items-baseline justify-between border-b border-slate-100 pb-1">
                <span className="text-slate-600 font-sans">Occupied Bandwidth (99% Power)</span>
                <span className="font-mono text-slate-900 font-medium">{formatHz(data.bandwidth)}</span>
              </div>

              <div className="flex items-baseline justify-between border-b border-slate-100 pb-1">
                <span className="text-slate-600 font-sans">Signal-to-Noise Ratio (SNR)</span>
                <span className="font-mono text-slate-900 font-medium">
                  {typeof data.snr === "number" ? data.snr.toFixed(1) : data.snr} dB
                </span>
              </div>

              <div className="flex items-baseline justify-between border-b border-slate-100 pb-1">
                <span className="text-slate-600 font-sans">Peak Frequency</span>
                <span className="font-mono text-slate-900 font-medium">{formatHz(data.peakFrequency)}</span>
              </div>

              <div className="flex items-baseline justify-between border-b border-slate-100 pb-1">
                <span className="text-slate-600 font-sans">Total Samples Processed</span>
                <span className="font-mono text-slate-900 font-medium">{formatCount(data.numSamples)}</span>
              </div>

              <div className="flex items-baseline justify-between border-b border-slate-100 pb-1">
                <span className="text-slate-600 font-sans">Modulation Format</span>
                <span className="font-mono text-slate-900 font-medium">{data.classification}</span>
              </div>

              <div className="flex items-baseline justify-between border-b border-slate-100 pb-1">
                <span className="text-slate-600 font-sans">Classification Confidence</span>
                <span className="font-mono text-slate-900 font-medium">{pct}%</span>
              </div>
            </div>
          </section>

          {/* Section 3: Signal Assessment */}
          <section>
            <div className="border-b border-slate-200 pb-1.5 mb-3">
              <h2 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                Signal Assessment Summary
              </h2>
            </div>
            <p className="text-xs text-slate-700 leading-relaxed font-sans">
              {data.explanation
                ? data.explanation.slice(0, 320) + (data.explanation.length > 320 ? "..." : "")
                : `Waveform capture ${sourceName} successfully processed through the SYNAPS DSP and modulation classification pipeline. Complete spectral characteristics and recovered data are documented on subsequent pages.`}
            </p>
          </section>
        </div>

        {/* Footer Page 1 */}
        <footer className="border-t border-slate-200 pt-3 text-[0.68rem] text-slate-500 flex justify-between items-center font-sans">
          <span>SYNAPS · Signal Intelligence</span>
          <span className="font-mono text-slate-400">Input: {sourceName}</span>
          <span>Page 1 / 3</span>
        </footer>
      </div>

      {/* ==================================================================== */}
      {/* PAGE 2: SIGNAL ANALYSIS & VISUALIZATIONS                            */}
      {/* ==================================================================== */}
      <div className="print-page flex flex-col justify-between">
        <div>
          {/* Running Header */}
          <header className="border-b border-slate-900 pb-2 mb-5 flex justify-between items-baseline">
            <h1 className="text-sm font-bold tracking-tight text-slate-950 uppercase font-sans">
              Signal Analysis & Visualizations
            </h1>
            <span className="text-xs text-slate-500 font-sans">SYNAPS · Technical Report</span>
          </header>

          {/* Extracted DSP & Statistical Features */}
          {data.features && data.features.length > 0 && (
            <section className="mb-6">
              <div className="border-b border-slate-200 pb-1 mb-2">
                <h2 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                  Extracted DSP & Statistical Features
                </h2>
              </div>

              <table className="w-full text-xs font-sans">
                <thead>
                  <tr className="border-b border-slate-300 text-slate-600 text-[0.7rem]">
                    <th className="py-1.5 text-left font-semibold">Feature</th>
                    <th className="py-1.5 text-right font-semibold">Measured Value</th>
                    <th className="py-1.5 text-left font-semibold pl-3">Unit</th>
                    <th className="py-1.5 text-left font-semibold">Significance</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {data.features.map((f) => (
                    <tr key={f.name}>
                      <td className="py-1.5 text-slate-800 font-medium">{f.name}</td>
                      <td className="py-1.5 text-right font-mono font-medium text-slate-950">{f.value}</td>
                      <td className="py-1.5 text-slate-500 font-mono pl-3">{f.unit || "—"}</td>
                      <td className="py-1.5 text-slate-600 text-[0.7rem]">{f.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          {/* Visualizations Section */}
          <section className="space-y-4">
            {/* Waveform */}
            <div>
              <div className="border-b border-slate-200 pb-1 mb-2">
                <h2 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                  Time-Domain Waveform
                </h2>
              </div>
              <PrintWaveformPlot samples={data.waveformSamples} time={data.waveformTime} />
            </div>

            {/* PSD & Spectrogram side-by-side */}
            <div>
              <div className="border-b border-slate-200 pb-1 mb-2">
                <h2 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                  Spectral Analysis
                </h2>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-[0.68rem] text-slate-600 mb-1 font-medium font-sans">
                    Power Spectral Density (PSD)
                  </p>
                  <PrintSpectrumPlot bins={data.spectrumBins} frequencies={data.spectrumFrequencies} />
                </div>

                <div>
                  <p className="text-[0.68rem] text-slate-600 mb-1 font-medium font-sans">
                    Time-Frequency Spectrogram
                  </p>
                  <PrintSpectrogramPlot
                    rows={data.spectrogramRows}
                    times={data.spectrogramTimes}
                    frequencies={data.spectrogramFrequencies}
                  />
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Footer Page 2 */}
        <footer className="border-t border-slate-200 pt-3 text-[0.68rem] text-slate-500 flex justify-between items-center font-sans">
          <span>SYNAPS · Signal Intelligence</span>
          <span className="font-mono text-slate-400">Input: {sourceName}</span>
          <span>Page 2 / 3</span>
        </footer>
      </div>

      {/* ==================================================================== */}
      {/* PAGE 3: SIGNAL RECOVERY & INTELLIGENCE                               */}
      {/* ==================================================================== */}
      <div className="print-page flex flex-col justify-between">
        <div>
          {/* Running Header */}
          <header className="border-b border-slate-900 pb-2 mb-5 flex justify-between items-baseline">
            <h1 className="text-sm font-bold tracking-tight text-slate-950 uppercase font-sans">
              Signal Recovery & Intelligence
            </h1>
            <span className="text-xs text-slate-500 font-sans">SYNAPS · Technical Report</span>
          </header>

          {/* Section 1: Data Recovery */}
          <section className="mb-6">
            <div className="border-b border-slate-200 pb-1 mb-3">
              <h2 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                Data Recovery & Ground-Truth Validation
              </h2>
            </div>

            <div className="grid grid-cols-5 gap-4 text-xs mb-3">
              <div className="border-b border-slate-100 pb-1.5">
                <span className="text-slate-500 font-sans block text-[0.68rem]">Reference Bits</span>
                <span className="font-mono text-slate-900 font-medium">
                  {data.bitRecovery?.reference_bit_count ?? "—"}
                </span>
              </div>

              <div className="border-b border-slate-100 pb-1.5">
                <span className="text-slate-500 font-sans block text-[0.68rem]">Recovered Bits</span>
                <span className="font-mono text-slate-900 font-medium">
                  {data.bitRecovery?.recovered_bit_count ?? (data.recoveredBitCount ? formatCount(data.recoveredBitCount) : "—")}
                </span>
              </div>

              <div className="border-b border-slate-100 pb-1.5">
                <span className="text-slate-500 font-sans block text-[0.68rem]">Bit Accuracy</span>
                <span className="font-mono text-slate-900 font-medium">
                  {data.bitRecovery?.bit_accuracy_pct !== undefined && data.bitRecovery?.bit_accuracy_pct !== null
                    ? `${data.bitRecovery.bit_accuracy_pct.toFixed(1)}%`
                    : "—"}
                </span>
              </div>

              <div className="border-b border-slate-100 pb-1.5">
                <span className="text-slate-500 font-sans block text-[0.68rem]">Bit Error Rate</span>
                <span className="font-mono text-slate-900 font-medium">
                  {data.bitRecovery?.ber !== undefined && data.bitRecovery?.ber !== null
                    ? data.bitRecovery.ber.toFixed(4)
                    : "—"}
                </span>
              </div>

              <div className="border-b border-slate-100 pb-1.5">
                <span className="text-slate-500 font-sans block text-[0.68rem]">Validation Status</span>
                <span className="font-sans text-slate-900 font-medium">
                  {data.bitRecovery?.validation_status ?? "Complete"}
                </span>
              </div>
            </div>

            {/* Decoded Message Payload if available */}
            {data.convertedData && (
              <div className="mt-3">
                <span className="text-[0.68rem] text-slate-500 font-sans uppercase block mb-1">
                  Decoded Message Payload ({data.dataEncoding || "ASCII"})
                </span>
                <pre className="p-2.5 bg-slate-50 border border-slate-200 rounded text-xs text-slate-900 font-mono whitespace-pre-wrap leading-relaxed">
                  {data.convertedData}
                </pre>
              </div>
            )}

            {/* Multi-carrier components table if mixed */}
            {data.signalType === "MIXED" && data.componentResults && data.componentResults.length > 0 && (
              <div className="mt-3">
                <span className="text-[0.68rem] text-slate-500 font-sans uppercase block mb-1">
                  Individual Carrier Component Separation
                </span>
                <table className="w-full text-xs font-sans border-t border-slate-200">
                  <thead>
                    <tr className="text-slate-500 text-[0.68rem] border-b border-slate-200">
                      <th className="py-1 text-left">#</th>
                      <th className="py-1 text-left">Modulation</th>
                      <th className="py-1 text-left">Symbol Rate</th>
                      <th className="py-1 text-left">Frequency Offset</th>
                      <th className="py-1 text-left">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono text-slate-800">
                    {data.componentResults.map((c) => (
                      <tr key={c.componentIndex}>
                        <td className="py-1">{c.componentIndex}</td>
                        <td className="py-1 font-sans font-medium text-slate-950">{c.modulation}</td>
                        <td className="py-1">{c.symbolRate ? `${(c.symbolRate / 1e3).toFixed(0)} kBaud` : "—"}</td>
                        <td className="py-1">{c.frequencyOffset ? `${(c.frequencyOffset / 1e3).toFixed(1)} kHz` : "0.0 kHz"}</td>
                        <td className="py-1 font-sans text-slate-600">{c.status}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {/* Section 2: RF & Advanced Identifiers */}
          <section className="mb-6">
            <div className="border-b border-slate-200 pb-1 mb-3">
              <h2 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                RF & Modulation Identifiers
              </h2>
            </div>

            <div className="grid grid-cols-3 gap-x-8 gap-y-2.5 text-xs">
              <div className="flex justify-between border-b border-slate-100 pb-1">
                <span className="text-slate-500 font-sans">Carrier Offset</span>
                <span className="font-mono text-slate-900 font-medium">{cfo}</span>
              </div>
              <div className="flex justify-between border-b border-slate-100 pb-1">
                <span className="text-slate-500 font-sans">Symbol Rate</span>
                <span className="font-mono text-slate-900 font-medium">{symbolRate}</span>
              </div>
              <div className="flex justify-between border-b border-slate-100 pb-1">
                <span className="text-slate-500 font-sans">PAPR</span>
                <span className="font-mono text-slate-900 font-medium">{papr}</span>
              </div>
              <div className="flex justify-between border-b border-slate-100 pb-1">
                <span className="text-slate-500 font-sans">Cumulant C40</span>
                <span className="font-mono text-slate-900 font-medium">{c40}</span>
              </div>
              <div className="flex justify-between border-b border-slate-100 pb-1">
                <span className="text-slate-500 font-sans">Cumulant C42</span>
                <span className="font-mono text-slate-900 font-medium">{c42}</span>
              </div>
              <div className="flex justify-between border-b border-slate-100 pb-1">
                <span className="text-slate-500 font-sans">RF Fingerprint</span>
                <span className="font-mono text-slate-900 font-medium truncate max-w-[120px]">{fingerprint}</span>
              </div>
            </div>
          </section>

          {/* Section 3: Technical Conclusion & Analysis Explanation */}
          {data.explanation && (
            <section className="mb-6">
              <div className="border-b border-slate-200 pb-1 mb-2.5">
                <h2 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                  Technical Assessment & Conclusion
                </h2>
              </div>
              <p className="text-xs text-slate-700 leading-relaxed font-sans text-justify">
                {data.explanation}
              </p>
            </section>
          )}
        </div>

        {/* Footer Page 3 */}
        <footer className="border-t border-slate-200 pt-3 text-[0.68rem] text-slate-500 flex justify-between items-center font-sans">
          <span>SYNAPS · Signal Intelligence</span>
          <span className="font-mono text-slate-400">Input: {sourceName}</span>
          <span>Page 3 / 3</span>
        </footer>
      </div>
    </div>
  );
}
