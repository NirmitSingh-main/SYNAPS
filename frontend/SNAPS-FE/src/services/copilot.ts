/**
 * SYNAPS Copilot — Frontend API service
 * Wraps POST /copilot/chat and never exposes GROQ_API_KEY to the browser.
 */

const API_BASE_URL = (import.meta.env["VITE_API_URL"] as string) || "";

export interface CopilotMessage {
  role: "user" | "assistant";
  content: string;
}

export interface CopilotChatRequest {
  message: string;
  page?: string;
  // Serialisable subset of AnalysisResponse (no binary arrays)
  analysis_context?: Record<string, unknown> | null;
  history?: CopilotMessage[];
}

export interface CopilotChatResponse {
  response: string;
  intent: string;
  model: string;
  page: string;
  success: boolean;
}

async function fetchCopilot(endpoint: string, options?: RequestInit): Promise<Response> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    return await fetch(url, options);
  } catch {
    if (!API_BASE_URL) {
      return await fetch(`http://localhost:8000${endpoint}`, options);
    }
    throw new Error("Network error — cannot reach SYNAPS backend.");
  }
}

export async function sendCopilotMessage(
  req: CopilotChatRequest,
  accessToken?: string | null
): Promise<CopilotChatResponse> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  if (accessToken) {
    headers["Authorization"] = `Bearer ${accessToken}`;
  }

  const res = await fetchCopilot("/copilot/chat", {
    method: "POST",
    headers,
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // ignore parse error
    }
    throw new Error(`Copilot error (${res.status}): ${detail}`);
  }

  return res.json() as Promise<CopilotChatResponse>;
}

/**
 * Build a serialisable analysis context from the full AnalysisResponse object,
 * stripping large binary arrays (waveform, spectrum, spectrogram) to keep the
 * payload small while preserving all fields relevant to the Copilot.
 */
export function buildAnalysisContext(
  analysis: Record<string, unknown> | null | undefined
): Record<string, unknown> | null {
  if (!analysis) return null;

  const {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    waveformSamples: _ws,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    waveformTime: _wt,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    spectrumBins: _sb,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    spectrumFrequencies: _sf,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    spectrogramRows: _sr,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    spectrogramTimes: _st,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    spectrogramFrequencies: _sfreq,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    raw_report: _rr,
    ...rest
  } = analysis;

  return rest as Record<string, unknown>;
}
