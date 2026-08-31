/**
 * API client service connecting SNAPS-FE frontend to SYNAPS AI & DSP Backend.
 */

const API_BASE_URL = (import.meta.env["VITE_API_URL"] as string) || "";

export interface FeatureItem {
  name: string;
  value: string;
  unit: string;
  description: string;
}

export interface PredictionItem {
  classLabel: string;
  probability: number;
}

export interface AnalysisResponse {
  id: string;
  timestamp: string;
  isDemoData: boolean;
  label: string;
  filename: string;
  format: string;
  classification: string;
  confidence: number;
  sampleRate: number;
  duration: number;
  bandwidth: number;
  snr: number;
  peakFrequency: number;
  numSamples: number;
  predictionBreakdown: PredictionItem[];
  features: FeatureItem[];
  explanation: string;
  waveformSamples: number[];
  spectrumBins: number[];
  spectrogramRows: number[][];
  raw_report?: any;
}

export interface SampleSignalItem {
  sample_id: string;
  filename: string;
  modulation: string;
  format: string;
  file_path: string;
  size_bytes: number;
}

export interface SampleListResponse {
  count: number;
  samples: SampleSignalItem[];
}

export interface HealthStatus {
  status: string;
  service: string;
  ai_available: boolean;
  version: string;
}

async function fetchWithFallback(endpoint: string, options?: RequestInit): Promise<Response> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, options);
    return res;
  } catch (err) {
    if (!API_BASE_URL) {
      return await fetch(`http://localhost:8000${endpoint}`, options);
    }
    throw err;
  }
}

/**
 * Check backend service health.
 */
export async function checkBackendHealth(): Promise<HealthStatus> {
  try {
    const res = await fetchWithFallback("/health");
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.warn("Backend health check failed:", err);
    return {
      status: "OFFLINE",
      service: "SYNAPS Backend",
      ai_available: false,
      version: "1.0.0",
    };
  }
}

/**
 * Fetch list of preloaded dataset sample signals from the backend.
 */
export async function getSampleSignals(): Promise<SampleSignalItem[]> {
  try {
    const res = await fetchWithFallback("/signal/samples");
    if (!res.ok) throw new Error(`Failed to list samples: ${res.statusText}`);
    const data: SampleListResponse = await res.json();
    return data.samples;
  } catch (err) {
    console.warn("Could not fetch samples from backend:", err);
    return [];
  }
}

/**
 * Upload an IQ or WAV file and execute full AI & DSP analysis.
 */
export async function uploadAndAnalyzeSignal(
  file: File,
  sampleRate?: number,
  samplesPerSymbol: number = 10
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (sampleRate) {
    formData.append("sample_rate", sampleRate.toString());
  }
  formData.append("samples_per_symbol", samplesPerSymbol.toString());

  const res = await fetchWithFallback("/analysis/upload-and-analyze", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Analysis failed (${res.status}): ${errText}`);
  }

  return await res.json();
}

/**
 * Analyze a preloaded server dataset sample by ID.
 */
export async function analyzeDatasetSample(
  sampleId: string,
  sampleRate?: number,
  samplesPerSymbol: number = 10
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("sample_id", sampleId);
  if (sampleRate) {
    formData.append("sample_rate", sampleRate.toString());
  }
  formData.append("samples_per_symbol", samplesPerSymbol.toString());

  const res = await fetchWithFallback("/analysis/sample", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Sample analysis failed (${res.status}): ${errText}`);
  }

  return await res.json();
}
