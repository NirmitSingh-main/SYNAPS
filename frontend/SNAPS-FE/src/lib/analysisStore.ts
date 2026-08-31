import { AnalysisResponse } from "@/services/api";
import { demoAnalysis } from "@/data/demoData";

interface AnalysisStore {
  file: File | null;
  sampleId: string | null;
  analysisId: string | null;
  analysisData: AnalysisResponse | null;
  error: string | null;
}

let store: AnalysisStore = {
  file: null,
  sampleId: null,
  analysisId: null,
  analysisData: null,
  error: null,
};

export function setAnalysisFile(file: File): void {
  store = {
    ...store,
    file,
    sampleId: null,
    analysisId: `session-${Date.now()}`,
    error: null,
  };
}

export function setAnalysisSample(sampleId: string): void {
  store = {
    ...store,
    file: null,
    sampleId,
    analysisId: `session-${Date.now()}`,
    error: null,
  };
}

export function setAnalysisData(data: AnalysisResponse): void {
  store = {
    ...store,
    analysisData: data,
    error: null,
  };
}

export function setAnalysisError(error: string): void {
  store = {
    ...store,
    error,
  };
}

export function getAnalysisFile(): File | null {
  return store.file;
}

export function getAnalysisSample(): string | null {
  return store.sampleId;
}

export function getAnalysisId(): string | null {
  return store.analysisId;
}

export function getAnalysisData(): AnalysisResponse | null {
  return store.analysisData;
}

export function getAnalysisError(): string | null {
  return store.error;
}

export function clearAnalysis(): void {
  store = {
    file: null,
    sampleId: null,
    analysisId: null,
    analysisData: null,
    error: null,
  };
}
