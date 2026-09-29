import { useEffect, useState, useMemo } from "react";
import { useNavigate } from "@tanstack/react-router";
import { ThemeProvider } from "@/lib/theme";
import { Header, Footer } from "@/components/si/Chrome";
import { useAuth, type Profile } from "@/lib/AuthContext";
import { supabase } from "@/lib/supabase";
import { setAnalysisData } from "@/lib/analysisStore";
import { type AnalysisResponse } from "@/services/api";
import { ModulationDonutChart } from "@/components/analytics/ModulationDonutChart";
import { ActivityLineChart, type ActivityPoint } from "@/components/analytics/ActivityLineChart";
import { ConfidenceBarChart } from "@/components/analytics/ConfidenceBarChart";
import { SignalQualityChart } from "@/components/analytics/SignalQualityChart";
import { Users, Activity, ShieldCheck, Radio, ShieldAlert } from "lucide-react";

interface AdminAnalysisRecord {
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
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days === 1) return "Yesterday";
  if (days < 30) return `${days}d ago`;
  return formatDate(dateStr);
}

const CANONICAL_MODS = ["BPSK", "QPSK", "FSK", "QAM16"];

export function AdminPage() {
  const { user, loading, isAdmin } = useAuth();
  const navigate = useNavigate();

  const [totalUsersCount, setTotalUsersCount] = useState<number | null>(null);
  const [totalAnalysesCount, setTotalAnalysesCount] = useState<number | null>(null);
  const [allReports, setAllReports] = useState<AdminAnalysisRecord[]>([]);
  const [recentUsers, setRecentUsers] = useState<Profile[]>([]);
  const [fetching, setFetching] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Access Control Guard
  useEffect(() => {
    if (loading) return;
    if (!user) {
      navigate({ to: "/auth" });
      return;
    }
    if (!isAdmin) {
      navigate({ to: "/dashboard" });
      return;
    }

    let isMounted = true;

    async function loadAdminPlatformData() {
      try {
        setFetching(true);
        const [usersRes, analysesCountRes, reportsRes, profilesRes] = await Promise.all([
          supabase.from("profiles").select("id", { count: "exact", head: true }),
          supabase.from("analysis_reports").select("id", { count: "exact", head: true }),
          supabase
            .from("analysis_reports")
            .select("*")
            .order("created_at", { ascending: false })
            .limit(100),
          supabase
            .from("profiles")
            .select("*")
            .order("created_at", { ascending: false })
            .limit(20),
        ]);

        if (isMounted) {
          setTotalUsersCount(usersRes.count ?? 0);
          setTotalAnalysesCount(analysesCountRes.count ?? reportsRes.data?.length ?? 0);
          setAllReports((reportsRes.data ?? []) as AdminAnalysisRecord[]);
          setRecentUsers((profilesRes.data ?? []) as Profile[]);
        }
      } catch (err) {
        console.error("Admin data fetch error:", err);
        if (isMounted) setError("Failed to fetch platform metrics.");
      } finally {
        if (isMounted) setFetching(false);
      }
    }

    loadAdminPlatformData();

    return () => {
      isMounted = false;
    };
  }, [user, loading, isAdmin, navigate]);

  const summary = useMemo(() => {
    const totalAnalyses = totalAnalysesCount ?? allReports.length;
    if (allReports.length === 0) {
      return {
        totalUsers: totalUsersCount ?? 0,
        totalAnalyses,
        avgConfidence: null,
        activeModulationsCount: 0,
      };
    }

    const confSum = allReports.reduce((acc, r) => acc + (r.confidence || 0), 0);
    const avgConfidence = (confSum / allReports.length) * 100;

    const uniqueMods = new Set(
      allReports.map((r) => (r.classification || "").trim().toUpperCase()).filter(Boolean)
    );

    return {
      totalUsers: totalUsersCount ?? recentUsers.length,
      totalAnalyses,
      avgConfidence,
      activeModulationsCount: uniqueMods.size,
    };
  }, [totalUsersCount, totalAnalysesCount, allReports, recentUsers]);

  const modulationDistribution = useMemo(() => {
    if (allReports.length === 0) return [];
    const counts: Record<string, number> = {};
    CANONICAL_MODS.forEach((m) => {
      counts[m] = 0;
    });

    allReports.forEach((r) => {
      const cls = r.classification || "Other";
      counts[cls] = (counts[cls] || 0) + 1;
    });

    return Object.entries(counts)
      .map(([label, count]) => ({ label, count }))
      .sort((a, b) => b.count - a.count);
  }, [allReports]);

  const activityData = useMemo<ActivityPoint[]>(() => {
    if (allReports.length === 0) return [];

    const dateMap: Record<string, number> = {};
    const sorted = [...allReports].sort(
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
  }, [allReports]);

  const confidenceByModulation = useMemo(() => {
    return CANONICAL_MODS.map((mod) => {
      const matching = allReports.filter(
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
  }, [allReports]);

  const snrByModulation = useMemo(() => {
    return CANONICAL_MODS.map((mod) => {
      const matching = allReports.filter(
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
  }, [allReports]);

  const userMap = useMemo(() => {
    const map: Record<string, string> = {};
    recentUsers.forEach((u) => {
      map[u.id] = u.email;
    });
    return map;
  }, [recentUsers]);

  const handleViewReport = (rep: AdminAnalysisRecord) => {
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

  if (loading || !user || !isAdmin) {
    return null;
  }

  return (
    <ThemeProvider>
      <div className="relative min-h-screen bg-background text-foreground flex flex-col justify-between">
        <Header />

        <main className="mx-auto w-full max-w-6xl px-4 sm:px-6 pb-32 pt-28">
          {/* Header */}
          <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <p className="label-mono">ADMIN / SIGNAL INTELLIGENCE</p>
              <h1 className="mt-2 text-2xl sm:text-3xl font-normal tracking-[-0.02em] text-foreground">
                Platform Overview
              </h1>
              <p className="mt-1 text-xs sm:text-sm text-muted-foreground">
                Platform-wide aggregated metrics, active modulations, and user registrations.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 border border-signal/60 bg-signal/15 px-3 py-1 font-mono text-[0.68rem] tracking-widest text-signal font-semibold rounded-md">
                <ShieldAlert className="h-3.5 w-3.5" />
                ADMINISTRATOR ACCESS
              </span>
            </div>
          </div>

          <div className="rule-line mb-8" />

          {error && (
            <div className="mb-8 floating-surface p-4 font-mono text-xs text-destructive border-destructive/50 bg-destructive/10">
              {error}
            </div>
          )}

          {/* 4 Summary Cards */}
          <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {/* Card 1: Total Users */}
            <div className="floating-surface floating-surface-hover p-5 transition-all">
              <div className="flex items-center justify-between">
                <p className="label-mono">TOTAL USERS</p>
                <Users className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="mt-3">
                <p className="font-mono text-3xl font-bold tracking-tight text-foreground">
                  {fetching ? (
                    <span className="inline-block h-8 w-16 animate-pulse bg-border/60" />
                  ) : (
                    summary.totalUsers
                  )}
                </p>
                <p className="mt-1 font-mono text-[0.68rem] text-muted-foreground">
                  Registered platform accounts
                </p>
              </div>
            </div>

            {/* Card 2: Total Analyses */}
            <div className="floating-surface floating-surface-hover p-5 transition-all">
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
                  Across all platform users
                </p>
              </div>
            </div>

            {/* Card 3: Avg Confidence */}
            <div className="floating-surface floating-surface-hover p-5 transition-all">
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
                  Platform-wide average
                </p>
              </div>
            </div>

            {/* Card 4: Active Modulations */}
            <div className="floating-surface floating-surface-hover p-5 transition-all">
              <div className="flex items-center justify-between">
                <p className="label-mono">ACTIVE MODULATIONS</p>
                <Radio className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="mt-3">
                <p className="font-mono text-3xl font-bold tracking-tight text-foreground">
                  {fetching ? (
                    <span className="inline-block h-8 w-16 animate-pulse bg-border/60" />
                  ) : (
                    summary.activeModulationsCount
                  )}
                </p>
                <p className="mt-1 font-mono text-[0.68rem] text-muted-foreground">
                  Distinct detected formats
                </p>
              </div>
            </div>
          </div>

          {/* Charts Row 1: Donut & Line */}
          <div className="mb-8 grid grid-cols-1 gap-6 lg:grid-cols-12">
            <div className="lg:col-span-6">
              <ModulationDonutChart
                data={modulationDistribution}
                title="Platform Modulation Distribution"
                subtitle="All user classifications breakdown"
                totalLabel="TOTAL"
              />
            </div>

            <div className="lg:col-span-6">
              <ActivityLineChart
                data={activityData}
                title="Platform Analysis Activity"
                subtitle="Aggregated daily analyses"
              />
            </div>
          </div>

          {/* Charts Row 2: Confidence & Signal Quality */}
          <div className="mb-8 grid grid-cols-1 gap-6 lg:grid-cols-12">
            <div className="lg:col-span-6">
              <ConfidenceBarChart
                data={confidenceByModulation}
                title="Average Confidence by Modulation"
                subtitle="Platform AI confidence averages"
              />
            </div>

            <div className="lg:col-span-6">
              <SignalQualityChart
                data={snrByModulation}
                title="Platform Signal Quality"
                subtitle="Platform average SNR by detected modulation"
              />
            </div>
          </div>

          {/* Tables Grid: Recent Analyses & Registered Users */}
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
            {/* Recent Analyses */}
            <div className="floating-surface p-5 lg:col-span-7">
              <p className="label-mono mb-1">RECENT PLATFORM ANALYSES</p>
              <p className="text-xs text-muted-foreground mb-4">
                Latest characterizations across all users
              </p>

              {fetching ? (
                <div className="space-y-2">
                  {[...Array(6)].map((_, i) => (
                    <div key={i} className="h-10 animate-pulse rounded-lg bg-border/40" />
                  ))}
                </div>
              ) : allReports.length === 0 ? (
                <p className="font-mono text-xs text-muted-foreground py-6 text-center">
                  No analyses recorded on platform yet.
                </p>
              ) : (
                <div className="space-y-1.5">
                  {allReports.slice(0, 8).map((r) => (
                    <div
                      key={r.id}
                      onClick={() => handleViewReport(r)}
                      className="floating-row flex cursor-pointer items-center justify-between px-3.5 py-2.5 font-mono text-xs"
                    >
                      <div className="flex items-center gap-2.5 min-w-0">
                        <span className="text-[0.68rem] text-muted-foreground shrink-0 max-w-[80px] truncate">
                          {userMap[r.user_id]?.split("@")[0] || `${r.user_id.slice(0, 5)}...`}
                        </span>
                        <span className="font-medium text-foreground truncate max-w-[120px] sm:max-w-[160px]">
                          {r.filename}
                        </span>
                        <span className="rounded-sm border border-signal/30 bg-signal/10 px-1.5 py-0.5 text-[0.65rem] font-semibold text-signal shrink-0">
                          {r.classification}
                        </span>
                      </div>

                      <div className="flex items-center gap-4 shrink-0 font-mono text-[0.72rem]">
                        <span className="text-foreground">
                          {(r.confidence * 100).toFixed(1)}%
                        </span>
                        <span className="text-muted-foreground hidden sm:inline">
                          {typeof r.snr === "number" ? `${r.snr.toFixed(1)} dB` : "—"}
                        </span>
                        <span className="text-muted-foreground text-[0.68rem]">
                          {timeAgo(r.created_at)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Registered Users */}
            <div className="floating-surface p-5 lg:col-span-5">
              <p className="label-mono mb-1">RECENT REGISTERED USERS</p>
              <p className="text-xs text-muted-foreground mb-4">
                User accounts & role authorizations
              </p>

              {fetching ? (
                <div className="space-y-2">
                  {[...Array(6)].map((_, i) => (
                    <div key={i} className="h-10 animate-pulse rounded-lg bg-border/40" />
                  ))}
                </div>
              ) : recentUsers.length === 0 ? (
                <p className="font-mono text-xs text-muted-foreground py-6 text-center">
                  No registered users found.
                </p>
              ) : (
                <div className="space-y-1.5">
                  {recentUsers.map((u) => (
                    <div
                      key={u.id}
                      className="floating-row flex items-center justify-between px-3.5 py-2.5 font-mono text-xs"
                    >
                      <span className="font-medium text-foreground truncate max-w-[150px]">
                        {u.email}
                      </span>
                      <div className="flex items-center gap-3 shrink-0">
                        <span
                          className={`rounded-sm px-1.5 py-0.5 text-[0.65rem] font-semibold uppercase ${
                            u.role === "admin"
                              ? "border border-signal/40 bg-signal/15 text-signal"
                              : "border border-border bg-surface text-muted-foreground"
                          }`}
                        >
                          {u.role}
                        </span>
                        <span className="text-[0.68rem] text-muted-foreground">
                          {timeAgo(u.created_at)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </main>

        <Footer />
      </div>
    </ThemeProvider>
  );
}
