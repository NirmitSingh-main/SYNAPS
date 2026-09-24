import { useEffect, useState, useMemo } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import { ThemeProvider } from "@/lib/theme";
import { Header, Footer } from "@/components/si/Chrome";
import { useAuth } from "@/lib/AuthContext";
import { supabase } from "@/lib/supabase";
import { setAnalysisData } from "@/lib/analysisStore";
import { type AnalysisResponse } from "@/services/api";
import { Search, Filter, ArrowUpDown, Radio, ArrowRight } from "lucide-react";

interface DBAnalysisRecord {
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
    year: "numeric",
  });
}

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days} days ago`;
  return formatDate(dateStr);
}

const MODULATION_FILTERS = ["ALL", "BPSK", "QPSK", "FSK", "QAM16"];

export function HistoryPage() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();

  const [records, setRecords] = useState<DBAnalysisRecord[]>([]);
  const [fetching, setFetching] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters and search
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedMod, setSelectedMod] = useState("ALL");
  const [sortBy, setSortBy] = useState<"newest" | "oldest" | "confidence" | "snr">("newest");

  useEffect(() => {
    if (!loading && !user) {
      navigate({ to: "/auth" });
      return;
    }
    if (!user) return;

    let isMounted = true;

    supabase
      .from("analysis_reports")
      .select("*")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false })
      .limit(30)
      .then(({ data, error: err }) => {
        if (!isMounted) return;
        if (err) {
          setError("Could not load analysis history.");
          console.error("History fetch error:", err);
        } else {
          setRecords((data ?? []) as DBAnalysisRecord[]);
        }
        setFetching(false);
      });

    return () => {
      isMounted = false;
    };
  }, [user, loading, navigate]);

  // Filtered & Sorted records
  const filteredRecords = useMemo(() => {
    return records
      .filter((rec) => {
        const matchesSearch =
          searchQuery.trim() === "" ||
          rec.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
          rec.classification.toLowerCase().includes(searchQuery.toLowerCase());

        const matchesMod =
          selectedMod === "ALL" ||
          rec.classification.toUpperCase() === selectedMod.toUpperCase();

        return matchesSearch && matchesMod;
      })
      .sort((a, b) => {
        if (sortBy === "newest") {
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        }
        if (sortBy === "oldest") {
          return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
        }
        if (sortBy === "confidence") {
          return b.confidence - a.confidence;
        }
        if (sortBy === "snr") {
          return (b.snr || 0) - (a.snr || 0);
        }
        return 0;
      });
  }, [records, searchQuery, selectedMod, sortBy]);

  const handleViewRecord = (rec: DBAnalysisRecord) => {
    const raw = rec.raw_report || {};
    const analysisResponse: AnalysisResponse = {
      id: rec.id,
      timestamp: rec.created_at,
      isDemoData: false,
      label: rec.classification,
      filename: rec.filename,
      format: rec.format,
      classification: rec.classification,
      confidence: rec.confidence,
      sampleRate: rec.sample_rate,
      duration: rec.duration,
      bandwidth: rec.bandwidth,
      snr: rec.snr,
      peakFrequency: rec.peak_frequency,
      numSamples: rec.num_samples,
      predictionBreakdown:
        rec.prediction_breakdown ||
        raw.predictionBreakdown || [
          { classLabel: rec.classification, probability: rec.confidence },
        ],
      features: rec.features || raw.features || [],
      explanation: rec.explanation || raw.explanation || "",
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

  if (loading || (!user && fetching)) return null;

  return (
    <ThemeProvider>
      <div className="relative min-h-screen bg-background text-foreground flex flex-col justify-between">
        <Header />

        <main className="mx-auto w-full max-w-5xl px-4 sm:px-6 pb-32 pt-28">
          {/* Header */}
          <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="label-mono">ANALYSIS HISTORY</p>
              <h1 className="mt-2 text-2xl sm:text-3xl font-normal tracking-[-0.02em] text-foreground">
                Your Signal Analyses
              </h1>
              <p className="mt-1 text-xs sm:text-sm text-muted-foreground">
                Showing your latest 30 characterization reports, newest first.
              </p>
            </div>

            <Link
              to="/analyze"
              className="hover-arrow inline-flex items-center gap-2 border border-signal/60 bg-signal/10 px-4 py-2 font-mono text-xs text-foreground transition-all hover:border-signal hover:bg-signal/20 shrink-0"
            >
              <span>Analyze Signal</span>
              <span className="arrow text-signal">→</span>
            </Link>
          </div>

          <div className="rule-line mb-8" />

          {/* Controls Bar: Search & Filters */}
          <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            {/* Search Box */}
            <div className="relative w-full sm:w-72">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search by filename or modulation..."
                className="w-full border border-border bg-surface/40 py-2 pl-9 pr-4 font-mono text-xs text-foreground placeholder:text-muted-foreground/60 focus:border-border-strong focus:outline-none"
              />
            </div>

            {/* Modulation Filter Pills & Sort Dropdown */}
            <div className="flex flex-wrap items-center gap-2 sm:gap-3">
              <div className="flex items-center gap-1 overflow-x-auto py-1">
                {MODULATION_FILTERS.map((mod) => (
                  <button
                    key={mod}
                    type="button"
                    onClick={() => setSelectedMod(mod)}
                    className={`px-2.5 py-1 font-mono text-[0.7rem] transition-colors rounded-xs ${
                      selectedMod === mod
                        ? "bg-signal text-primary-foreground font-semibold"
                        : "border border-border bg-surface/30 text-muted-foreground hover:border-border-strong hover:text-foreground"
                    }`}
                  >
                    {mod}
                  </button>
                ))}
              </div>

              {/* Sort Selector */}
              <div className="flex items-center gap-1.5 border border-border bg-surface/30 px-2.5 py-1">
                <ArrowUpDown className="h-3 w-3 text-muted-foreground" />
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                  className="bg-transparent font-mono text-[0.7rem] text-foreground focus:outline-none cursor-pointer"
                >
                  <option value="newest" className="bg-background text-foreground">
                    Newest First
                  </option>
                  <option value="oldest" className="bg-background text-foreground">
                    Oldest First
                  </option>
                  <option value="confidence" className="bg-background text-foreground">
                    Highest Confidence
                  </option>
                  <option value="snr" className="bg-background text-foreground">
                    Highest SNR
                  </option>
                </select>
              </div>
            </div>
          </div>

          {/* Records List / Table */}
          {fetching ? (
            <div className="space-y-3">
              {[...Array(6)].map((_, i) => (
                <div
                  key={i}
                  className="h-16 animate-pulse border border-border bg-surface/30"
                />
              ))}
            </div>
          ) : error ? (
            <div className="border border-destructive/40 bg-destructive/10 p-4 font-mono text-xs text-destructive">
              {error}
            </div>
          ) : records.length === 0 ? (
            <div className="border border-dashed border-border/80 bg-surface/20 px-6 py-20 text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full border border-border bg-background">
                <Radio className="h-5 w-5 text-signal" />
              </div>
              <h3 className="mt-4 text-base font-medium text-foreground">
                No analyses yet
              </h3>
              <p className="mt-1 text-xs text-muted-foreground">
                Run an analysis to start building your history.
              </p>
              <div className="mt-6">
                <Link
                  to="/analyze"
                  className="hover-arrow inline-flex items-center gap-2 border border-border-strong bg-surface px-5 py-2.5 font-mono text-xs text-foreground transition-all hover:border-signal"
                >
                  <span>Analyze Signal</span>
                  <span className="arrow text-signal">→</span>
                </Link>
              </div>
            </div>
          ) : filteredRecords.length === 0 ? (
            <div className="border border-dashed border-border px-6 py-12 text-center">
              <p className="font-mono text-xs text-muted-foreground">
                No analyses match the filter "{selectedMod}" or search query "{searchQuery}".
              </p>
              <button
                type="button"
                onClick={() => {
                  setSearchQuery("");
                  setSelectedMod("ALL");
                }}
                className="mt-3 font-mono text-xs text-signal underline hover:text-signal/80"
              >
                Clear filters
              </button>
            </div>
          ) : (
            <div className="border border-border bg-surface/30">
              {/* Desktop View Table */}
              <div className="hidden md:block overflow-x-auto">
                <table className="w-full text-left font-mono text-xs">
                  <thead>
                    <tr className="border-b border-border bg-background/50 text-[0.7rem] text-muted-foreground">
                      <th className="py-3 px-4 font-normal">FILENAME</th>
                      <th className="py-3 px-4 font-normal">MODULATION</th>
                      <th className="py-3 px-4 font-normal">CONFIDENCE</th>
                      <th className="py-3 px-4 font-normal">SNR</th>
                      <th className="py-3 px-4 font-normal">DATE</th>
                      <th className="py-3 px-4 text-right font-normal">ACTION</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/40">
                    {filteredRecords.map((rec) => (
                      <tr
                        key={rec.id}
                        onClick={() => handleViewRecord(rec)}
                        className="group cursor-pointer transition-colors hover:bg-surface/80"
                      >
                        <td className="py-3.5 px-4 font-medium text-foreground">
                          <span className="block max-w-[240px] truncate">
                            {rec.filename}
                          </span>
                        </td>
                        <td className="py-3.5 px-4">
                          <span className="rounded-sm border border-signal/30 bg-signal/10 px-2 py-0.5 font-mono text-[0.7rem] text-signal font-semibold">
                            {rec.classification}
                          </span>
                        </td>
                        <td className="py-3.5 px-4 text-foreground">
                          {(rec.confidence * 100).toFixed(1)}%
                        </td>
                        <td className="py-3.5 px-4 text-muted-foreground">
                          {typeof rec.snr === "number" ? `${rec.snr.toFixed(1)} dB` : "—"}
                        </td>
                        <td className="py-3.5 px-4 text-muted-foreground">
                          {timeAgo(rec.created_at)}
                        </td>
                        <td className="py-3.5 px-4 text-right">
                          <span className="inline-flex items-center gap-1 font-mono text-[0.72rem] text-signal font-medium group-hover:underline">
                            View <span className="arrow font-mono">→</span>
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Mobile Card List View */}
              <div className="divide-y divide-border/40 md:hidden">
                {filteredRecords.map((rec) => (
                  <div
                    key={rec.id}
                    onClick={() => handleViewRecord(rec)}
                    className="cursor-pointer p-4 transition-colors hover:bg-surface/80"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate font-mono text-xs font-semibold text-foreground">
                          {rec.filename}
                        </p>
                        <div className="mt-1.5 flex flex-wrap items-center gap-2">
                          <span className="rounded-sm border border-signal/30 bg-signal/10 px-1.5 py-0.5 font-mono text-[0.65rem] text-signal font-semibold">
                            {rec.classification}
                          </span>
                          <span className="font-mono text-[0.68rem] text-foreground">
                            {(rec.confidence * 100).toFixed(1)}%
                          </span>
                          <span className="text-muted-foreground">·</span>
                          <span className="font-mono text-[0.68rem] text-muted-foreground">
                            {typeof rec.snr === "number" ? `${rec.snr.toFixed(1)} dB` : "—"}
                          </span>
                        </div>
                      </div>

                      <div className="text-right shrink-0">
                        <span className="block font-mono text-[0.65rem] text-muted-foreground">
                          {timeAgo(rec.created_at)}
                        </span>
                        <span className="mt-2 inline-flex items-center gap-1 font-mono text-xs text-signal font-medium">
                          View →
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </main>

        <Footer />
      </div>
    </ThemeProvider>
  );
}
