import { useEffect, useState, useMemo } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import { ThemeProvider } from "@/lib/theme";
import { Header, Footer } from "@/components/si/Chrome";
import { useAuth } from "@/lib/AuthContext";
import { supabase } from "@/lib/supabase";
import { setAnalysisData } from "@/lib/analysisStore";
import { type AnalysisResponse } from "@/services/api";
import { ModulationDonutChart } from "@/components/analytics/ModulationDonutChart";
import { ActivityLineChart, type ActivityPoint } from "@/components/analytics/ActivityLineChart";
import { ConfidenceBarChart } from "@/components/analytics/ConfidenceBarChart";
import { SignalQualityChart } from "@/components/analytics/SignalQualityChart";
import {
  Activity,
  Zap,
  Radio,
  BarChart3,
  ArrowRight,
  TrendingUp,
  ShieldCheck,
  Clock,
  Sparkles,
} from "lucide-react";

export interface DBReport {
  id: string;
  user_id: string;
  created_at: string;
  filename: string;
  format: string;
  classification: string;
  confidence: number;
  sample_rate: number;
  duration: number;
  bandwidth: number;
  snr: number;
  peak_frequency: number;
  num_samples: number;
  prediction_breakdown: any;
  features: any;
  explanation: string;
  raw_report: any;
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days === 1) return "Yesterday";
  if (days < 30) return `${days}d ago`;
  return formatDate(dateStr);
}

const CANONICAL_MODS = ["BPSK", "QPSK", "FSK", "QAM16"];

export function DashboardPage() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();

  const [reports, setReports] = useState<DBReport[]>([]);
  const [fetching, setFetching] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Auth Guard
  useEffect(() => {
    if (!loading && !user) {
      navigate({ to: "/auth" });
    }
  }, [user, loading, navigate]);

  // Fetch Reports
  useEffect(() => {
    if (!user) return;

    let isMounted = true;
    async function loadData() {
      try {
        setFetching(true);
        const { data, error: err } = await supabase
          .from("analysis_reports")
          .select("*")
          .eq("user_id", user!.id)
          .order("created_at", { ascending: false });

        if (err) {
          console.error("Dashboard reports fetch error:", err);
          if (isMounted) setError("Failed to load your analysis records.");
        } else if (isMounted) {
          setReports((data ?? []) as DBReport[]);
        }
      } catch (err) {
        console.error("Dashboard error:", err);
        if (isMounted) setError("An unexpected error occurred while loading dashboard.");
      } finally {
        if (isMounted) setFetching(false);
      }
    }

    loadData();
    return () => {
      isMounted = false;
    };
  }, [user]);

  interface SummaryData {
    totalAnalyses: number;
    avgConfidence: number | null;
    mostDetected: string | null;
    avgSnr: number | null;
    highestConfidenceReport: DBReport | null;
    latestReport: DBReport | null;
  }

  // Computed summary metrics
  const summary = useMemo<SummaryData>(() => {
    const totalAnalyses = reports.length;

    if (totalAnalyses === 0) {
      return {
        totalAnalyses: 0,
        avgConfidence: null,
        mostDetected: null,
        avgSnr: null,
        highestConfidenceReport: null,
        latestReport: null,
      };
    }

    // Confidence sum
    const totalConf = reports.reduce((acc, r) => acc + (r.confidence || 0), 0);
    const avgConfidence = (totalConf / totalAnalyses) * 100;

    // SNR sum
    const validSnrReports = reports.filter(
      (r) => typeof r.snr === "number" && !isNaN(r.snr)
    );
    const totalSnr = validSnrReports.reduce((acc, r) => acc + r.snr, 0);
    const avgSnr = validSnrReports.length > 0 ? totalSnr / validSnrReports.length : null;

    // Most detected
    const counts: Record<string, number> = {};
    reports.forEach((r) => {
      const cls = r.classification || "Unknown";
      counts[cls] = (counts[cls] || 0) + 1;
    });

    let topMod: string | null = null;
    let topCount = -1;
    Object.entries(counts).forEach(([mod, count]) => {
      if (count > topCount) {
        topCount = count;
        topMod = mod;
      }
    });

    // Highest confidence report
    let highestConf = -1;
    let highestReport: DBReport | null = null;
    reports.forEach((r) => {
      if (r.confidence > highestConf) {
        highestConf = r.confidence;
        highestReport = r;
      }
    });

    return {
      totalAnalyses,
      avgConfidence,
      mostDetected: topMod,
      avgSnr,
      highestConfidenceReport: highestReport,
      latestReport: reports[0] || null,
    };
  }, [reports]);

  // Modulation distribution
  const modulationDistribution = useMemo(() => {
    if (reports.length === 0) return [];
    const counts: Record<string, number> = {};
    CANONICAL_MODS.forEach((m) => {
      counts[m] = 0;
    });

    reports.forEach((r) => {
      const cls = r.classification || "Other";
      counts[cls] = (counts[cls] || 0) + 1;
    });

    return Object.entries(counts)
      .map(([label, count]) => ({ label, count }))
      .sort((a, b) => b.count - a.count);
  }, [reports]);

  // Activity over time (e.g. daily/recent bucket)
  const activityData = useMemo<ActivityPoint[]>(() => {
    if (reports.length === 0) return [];

    // Group by date (YYYY-MM-DD)
    const dateMap: Record<string, number> = {};

    // Sort ascending for chronology
    const sorted = [...reports].sort(
      (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    );

    sorted.forEach((r) => {
      const d = new Date(r.created_at);
      const key = d.toISOString().split("T")[0];
      if (key) {
        dateMap[key] = (dateMap[key] || 0) + 1;
      }
    });

    return Object.entries(dateMap).map(([key, count]) => {
      const parts = key.split("-");
      const year = parseInt(parts[0] || "2026", 10);
      const month = parseInt(parts[1] || "1", 10) - 1;
      const day = parseInt(parts[2] || "1", 10);
      const d = new Date(year, month, day);
      const dateLabel = d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
      return {
        date: dateLabel,
        fullDate: d.toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
          year: "numeric",
        }),
        count,
      };
    });
  }, [reports]);

  // Average confidence per modulation
  const confidenceByModulation = useMemo(() => {
    return CANONICAL_MODS.map((mod) => {
      const matching = reports.filter(
        (r) => (r.classification || "").toUpperCase() === mod.toUpperCase()
      );
      if (matching.length === 0) {
        return { label: mod, avgConfidence: null, sampleCount: 0 };
      }
      const sum = matching.reduce((acc, r) => acc + (r.confidence || 0), 0);
      return {
        label: mod,
        avgConfidence: (sum / matching.length) * 100,
        sampleCount: matching.length,
      };
    });
  }, [reports]);

  // Signal quality (average SNR) per modulation
  const snrByModulation = useMemo(() => {
    return CANONICAL_MODS.map((mod) => {
      const matching = reports.filter(
        (r) =>
          (r.classification || "").toUpperCase() === mod.toUpperCase() &&
          typeof r.snr === "number" &&
          !isNaN(r.snr)
      );
      if (matching.length === 0) {
        return { label: mod, avgSnr: null, sampleCount: 0 };
      }
      const sum = matching.reduce((acc, r) => acc + r.snr, 0);
      return {
        label: mod,
        avgSnr: sum / matching.length,
        sampleCount: matching.length,
      };
    });
  }, [reports]);

  // Recent analyses (5 to 8 rows)
  const recentAnalyses = useMemo(() => {
    return reports.slice(0, 6);
  }, [reports]);

  // Handle row click to inspect result
  const handleViewReport = (rep: DBReport) => {
    const raw = rep.raw_report || {};
    const analysisResponse: AnalysisResponse = {
      id: rep.id,
      timestamp: rep.created_at,
      isDemoData: false,
      label: rep.classification,
      filename: rep.filename,
      format: rep.format,
      classification: rep.classification,
      confidence: rep.confidence,
      sampleRate: rep.sample_rate,
      duration: rep.duration,
      bandwidth: rep.bandwidth,
      snr: rep.snr,
      peakFrequency: rep.peak_frequency,
      numSamples: rep.num_samples,
      predictionBreakdown:
        rep.prediction_breakdown ||
        raw.predictionBreakdown || [
          { classLabel: rep.classification, probability: rep.confidence },
        ],
      features: rep.features || raw.features || [],
      explanation: rep.explanation || raw.explanation || "",
      recoveredBitCount: raw.recoveredBitCount ?? null,
      convertedData: raw.convertedData ?? null,
      decodingStatus: raw.decodingStatus ?? null,
      dataEncoding: raw.dataEncoding ?? "ASCII",
      dataConversionValid: raw.dataConversionValid ?? false,
      waveformSamples: raw.waveformSamples || [],
      spectrumBins: raw.spectrumBins || [],
      spectrogramRows: raw.spectrogramRows || [],
    };

    setAnalysisData(analysisResponse);
    navigate({ to: "/results" });
  };

  if (loading || (!user && fetching)) {
    return null;
  }

  return (
    <ThemeProvider>
      <div className="relative min-h-screen bg-background text-foreground flex flex-col justify-between">
        <Header />

        <main className="mx-auto w-full max-w-6xl px-4 sm:px-6 pb-32 pt-28">
          {/* ---- Header Section ---- */}
          <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <div className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-signal animate-pulse" />
                <p className="label-mono">SIGNAL INTELLIGENCE</p>
              </div>
              <h1 className="mt-2 text-2xl sm:text-3xl font-normal tracking-[-0.02em] text-foreground">
                Your Analysis Overview
              </h1>
              <p className="mt-1 text-xs sm:text-sm text-muted-foreground">
                A compact view of your recent RF signal analysis activity.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <Link
                to="/analyze"
                className="hover-arrow inline-flex items-center gap-2 border border-signal/60 bg-signal/10 px-4 py-2 font-mono text-xs text-foreground transition-all hover:border-signal hover:bg-signal/20"
              >
                <span>+ Analyze New Signal</span>
                <ArrowRight className="h-3 w-3 text-signal" />
              </Link>
            </div>
          </div>

          <div className="rule-line mb-8" />

          {error && (
            <div className="mb-8 border border-destructive/50 bg-destructive/10 p-4 font-mono text-xs text-destructive">
              {error}
            </div>
          )}

          {/* ---- Summary Cards (4 Columns) ---- */}
          <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {/* Card 1: Total Analyses */}
            <div className="border border-border bg-surface/30 p-5 transition-colors hover:border-border-strong">
              <div className="flex items-center justify-between">
                <p className="label-mono">TOTAL ANALYSES</p>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="mt-3">
                <p className="font-mono text-3xl font-bold tracking-tight text-foreground">
                  {fetching ? (
                    <span className="inline-block h-8 w-16 animate-pulse bg-border/60" />
                  ) : (
                    summary.totalAnalyses
                  )}
                </p>
                <p className="mt-1 font-mono text-[0.68rem] text-muted-foreground">
                  All-time analyses
                </p>
              </div>
            </div>

            {/* Card 2: Avg Confidence */}
            <div className="border border-border bg-surface/30 p-5 transition-colors hover:border-border-strong">
              <div className="flex items-center justify-between">
                <p className="label-mono">AVG CONFIDENCE</p>
                <ShieldCheck className="h-4 w-4 text-signal" />
              </div>
              <div className="mt-3">
                <p className="font-mono text-3xl font-bold tracking-tight text-foreground">
                  {fetching ? (
                    <span className="inline-block h-8 w-20 animate-pulse bg-border/60" />
                  ) : summary.avgConfidence !== null ? (
                    `${summary.avgConfidence.toFixed(1)}%`
                  ) : (
                    "—"
                  )}
                </p>
                <p className="mt-1 font-mono text-[0.68rem] text-muted-foreground">
                  Across analyzed signals
                </p>
              </div>
            </div>

            {/* Card 3: Most Detected */}
            <div className="border border-border bg-surface/30 p-5 transition-colors hover:border-border-strong">
              <div className="flex items-center justify-between">
                <p className="label-mono">MOST DETECTED</p>
                <Radio className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="mt-3">
                <p className="font-mono text-3xl font-bold tracking-tight text-foreground">
                  {fetching ? (
                    <span className="inline-block h-8 w-24 animate-pulse bg-border/60" />
                  ) : (
                    summary.mostDetected || "—"
                  )}
                </p>
                <p className="mt-1 font-mono text-[0.68rem] text-muted-foreground">
                  Most frequently classified
                </p>
              </div>
            </div>

            {/* Card 4: Avg SNR */}
            <div className="border border-border bg-surface/30 p-5 transition-colors hover:border-border-strong">
              <div className="flex items-center justify-between">
                <p className="label-mono">AVG SNR</p>
                <Zap className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="mt-3">
                <p className="font-mono text-3xl font-bold tracking-tight text-foreground">
                  {fetching ? (
                    <span className="inline-block h-8 w-20 animate-pulse bg-border/60" />
                  ) : summary.avgSnr !== null ? (
                    `${summary.avgSnr.toFixed(1)} dB`
                  ) : (
                    "—"
                  )}
                </p>
                <p className="mt-1 font-mono text-[0.68rem] text-muted-foreground">
                  Across analyzed signals
                </p>
              </div>
            </div>
          </div>

          {/* ---- Empty State if no analyses ---- */}
          {!fetching && reports.length === 0 && (
            <div className="mb-12 border border-dashed border-border/80 bg-surface/20 px-6 py-16 text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full border border-border bg-background">
                <Radio className="h-5 w-5 text-signal" />
              </div>
              <h3 className="mt-4 text-base font-medium text-foreground">
                No analyses yet
              </h3>
              <p className="mt-1 text-xs text-muted-foreground">
                Upload or analyze your first RF signal to generate metrics and insights.
              </p>
              <div className="mt-6">
                <Link
                  to="/analyze"
                  className="hover-arrow inline-flex items-center gap-2 border border-border-strong bg-surface px-5 py-2.5 font-mono text-xs text-foreground transition-all hover:border-signal"
                >
                  <span>Run your first analysis</span>
                  <span className="arrow text-signal">→</span>
                </Link>
              </div>
            </div>
          )}

          {/* ---- Main Analytics Grid (2 Columns) ---- */}
          <div className="mb-8 grid grid-cols-1 gap-6 lg:grid-cols-12">
            <div className="lg:col-span-6">
              {fetching ? (
                <div className="h-72 animate-pulse border border-border bg-surface/30 p-6" />
              ) : (
                <ModulationDonutChart
                  data={modulationDistribution}
                  title="Modulation Distribution"
                  subtitle="Classification breakdown across your analyses"
                  totalLabel="TOTAL"
                />
              )}
            </div>

            <div className="lg:col-span-6">
              {fetching ? (
                <div className="h-72 animate-pulse border border-border bg-surface/30 p-6" />
              ) : (
                <ActivityLineChart
                  data={activityData}
                  title="Analysis Activity"
                  subtitle="Your recent analysis frequency"
                />
              )}
            </div>
          </div>

          {/* ---- Second Analytics Row (2 Columns) ---- */}
          <div className="mb-8 grid grid-cols-1 gap-6 lg:grid-cols-12">
            <div className="lg:col-span-6">
              {fetching ? (
                <div className="h-64 animate-pulse border border-border bg-surface/30 p-6" />
              ) : (
                <ConfidenceBarChart
                  data={confidenceByModulation}
                  title="Average Confidence"
                  subtitle="Average AI confidence by modulation"
                />
              )}
            </div>

            <div className="lg:col-span-6">
              {fetching ? (
                <div className="h-64 animate-pulse border border-border bg-surface/30 p-6" />
              ) : (
                <SignalQualityChart
                  data={snrByModulation}
                  title="Signal Quality"
                  subtitle="Average SNR by detected modulation"
                />
              )}
            </div>
          </div>

          {/* ---- Quick Insights Strip ---- */}
          {!fetching && reports.length > 0 && (
            <div className="mb-8 border border-border bg-background/60 p-4">
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="h-3.5 w-3.5 text-signal" />
                <p className="label-mono">SIGNAL INSIGHTS</p>
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-4 font-mono text-xs">
                <div className="border-l-2 border-signal/60 pl-3">
                  <span className="text-muted-foreground block text-[0.65rem]">
                    PRIMARY MODULATION
                  </span>
                  <span className="font-semibold text-foreground text-sm">
                    {summary.mostDetected || "—"}
                  </span>
                </div>
                <div className="border-l-2 border-signal/60 pl-3">
                  <span className="text-muted-foreground block text-[0.65rem]">
                    HIGHEST CONFIDENCE
                  </span>
                  <span className="font-semibold text-foreground text-sm">
                    {summary.highestConfidenceReport
                      ? `${(summary.highestConfidenceReport.confidence * 100).toFixed(1)}% (${summary.highestConfidenceReport.classification})`
                      : "—"}
                  </span>
                </div>
                <div className="border-l-2 border-signal/60 pl-3">
                  <span className="text-muted-foreground block text-[0.65rem]">
                    OVERALL MEAN SNR
                  </span>
                  <span className="font-semibold text-foreground text-sm">
                    {summary.avgSnr !== null ? `${summary.avgSnr.toFixed(1)} dB` : "—"}
                  </span>
                </div>
                <div className="border-l-2 border-signal/60 pl-3">
                  <span className="text-muted-foreground block text-[0.65rem]">
                    LATEST ANALYSIS
                  </span>
                  <span className="font-semibold text-foreground text-sm">
                    {summary.latestReport ? timeAgo(summary.latestReport.created_at) : "—"}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* ---- Recent Analyses Section ---- */}
          <div className="border border-border bg-surface/30 p-5">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <p className="label-mono">RECENT ANALYSES</p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  Latest signal characterization records
                </p>
              </div>
              <Link
                to="/history"
                className="hover-arrow inline-flex items-center gap-1 font-mono text-xs text-signal transition-colors hover:underline"
              >
                <span>VIEW ALL</span>
                <span className="arrow font-mono">→</span>
              </Link>
            </div>

            {fetching ? (
              <div className="space-y-2">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="h-12 animate-pulse bg-border/40" />
                ))}
              </div>
            ) : recentAnalyses.length === 0 ? (
              <div className="py-8 text-center">
                <p className="font-mono text-xs text-muted-foreground">
                  No recent analyses to display.
                </p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs">
                  <thead>
                    <tr className="border-b border-border text-muted-foreground text-[0.7rem]">
                      <th className="py-2.5 pr-4 font-normal">FILENAME</th>
                      <th className="py-2.5 pr-4 font-normal">MODULATION</th>
                      <th className="py-2.5 pr-4 font-normal">CONFIDENCE</th>
                      <th className="py-2.5 pr-4 font-normal">SNR</th>
                      <th className="py-2.5 pr-4 font-normal">DATE</th>
                      <th className="py-2.5 text-right font-normal">ACTION</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/40">
                    {recentAnalyses.map((rec) => (
                      <tr
                        key={rec.id}
                        onClick={() => handleViewReport(rec)}
                        className="group cursor-pointer transition-colors hover:bg-surface/80"
                      >
                        <td className="py-3 pr-4 font-medium text-foreground">
                          <span className="block max-w-[220px] truncate">
                            {rec.filename}
                          </span>
                        </td>
                        <td className="py-3 pr-4">
                          <span className="rounded-sm border border-signal/30 bg-signal/10 px-2 py-0.5 font-mono text-[0.7rem] text-signal font-semibold">
                            {rec.classification}
                          </span>
                        </td>
                        <td className="py-3 pr-4 text-foreground">
                          {(rec.confidence * 100).toFixed(1)}%
                        </td>
                        <td className="py-3 pr-4 text-muted-foreground">
                          {typeof rec.snr === "number" ? `${rec.snr.toFixed(1)} dB` : "—"}
                        </td>
                        <td className="py-3 pr-4 text-muted-foreground">
                          {timeAgo(rec.created_at)}
                        </td>
                        <td className="py-3 text-right">
                          <span className="inline-flex items-center gap-1 text-[0.72rem] text-muted-foreground group-hover:text-signal">
                            View <span className="arrow font-mono text-signal">→</span>
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </main>

        <Footer />
      </div>
    </ThemeProvider>
  );
}
