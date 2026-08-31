import { useRef, useState, useCallback, type DragEvent, type KeyboardEvent } from "react";
import { Upload, X } from "lucide-react";

const ACCEPTED_EXTENSIONS = [".wav", ".iq"];
const MAX_SIZE_MB = 500;

function getExtension(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot >= 0 ? filename.slice(dot).toLowerCase() : "";
}

function isValidExtension(filename: string): boolean {
  return ACCEPTED_EXTENSIONS.includes(getExtension(filename));
}

interface FileUploaderProps {
  onFileSelected: (file: File) => void;
  onError?: (msg: string) => void;
}

export function FileUploader({ onFileSelected, onError }: FileUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      setError(null);
      if (!isValidExtension(file.name)) {
        const msg = `Unsupported file type "${getExtension(file.name)}". Please upload a .wav or .iq file.`;
        setError(msg);
        onError?.(msg);
        return;
      }
      const sizeMB = file.size / (1024 * 1024);
      if (sizeMB > MAX_SIZE_MB) {
        const msg = `File exceeds maximum size of ${MAX_SIZE_MB} MB.`;
        setError(msg);
        onError?.(msg);
        return;
      }
      onFileSelected(file);
    },
    [onFileSelected, onError]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    // Reset input so same file can be re-selected after removal
    e.target.value = "";
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e: DragEvent<HTMLDivElement>) => {
    // Only clear if leaving the drop zone itself, not a child
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragging(false);
    }
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const openPicker = () => inputRef.current?.click();

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openPicker();
    }
  };

  return (
    <div className="space-y-3">
      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        accept=".wav,.iq"
        onChange={onInputChange}
        className="sr-only"
        aria-label="Upload signal file"
        id="signal-file-input"
        tabIndex={-1}
      />

      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Drop zone — drag and drop a WAV or IQ file, or press Enter to browse"
        aria-describedby={error ? "upload-error" : undefined}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={openPicker}
        onKeyDown={onKeyDown}
        style={{
          borderColor: isDragging ? "var(--signal)" : "var(--border-strong)",
          backgroundColor: isDragging ? "color-mix(in oklab, var(--signal) 6%, var(--background))" : "var(--background)",
        }}
        className="relative flex min-h-[280px] cursor-pointer flex-col items-center justify-center gap-5 border border-dashed transition-all duration-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 sm:min-h-[320px]"
      >
        {/* Subtle corner markers */}
        <span aria-hidden className="pointer-events-none absolute top-3 left-3 h-4 w-4 border-t border-l border-border-strong opacity-60" />
        <span aria-hidden className="pointer-events-none absolute top-3 right-3 h-4 w-4 border-t border-r border-border-strong opacity-60" />
        <span aria-hidden className="pointer-events-none absolute bottom-3 left-3 h-4 w-4 border-b border-l border-border-strong opacity-60" />
        <span aria-hidden className="pointer-events-none absolute bottom-3 right-3 h-4 w-4 border-b border-r border-border-strong opacity-60" />

        {/* Upload icon */}
        <div
          style={{
            color: isDragging ? "var(--signal)" : "var(--muted-foreground)",
          }}
          className="transition-colors duration-300"
        >
          <Upload
            className="h-7 w-7 transition-transform duration-300"
            style={{ transform: isDragging ? "translateY(-3px)" : "none" }}
            strokeWidth={1.25}
          />
        </div>

        <div className="text-center">
          <p className="text-[0.9rem] text-foreground">
            {isDragging ? "Release to drop" : "Drop your signal"}
          </p>
          <p className="mt-2 text-sm text-muted-foreground">
            or{" "}
            <span className="underline underline-offset-2 transition-colors duration-200 hover:text-foreground">
              browse files
            </span>
          </p>
        </div>

        {/* Format label */}
        <div className="flex items-center gap-3">
          {ACCEPTED_EXTENSIONS.map((ext) => (
            <span
              key={ext}
              className="font-mono text-[0.68rem] tracking-[0.14em] text-muted-foreground"
            >
              {ext.toUpperCase().slice(1)}
            </span>
          ))}
          <span className="font-mono text-[0.68rem] text-border-strong" aria-hidden>
            ·
          </span>
          <span className="font-mono text-[0.68rem] tracking-[0.08em] text-muted-foreground">
            up to {MAX_SIZE_MB} MB
          </span>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div
          id="upload-error"
          role="alert"
          className="flex items-start gap-2 border border-destructive/40 bg-destructive/8 px-4 py-3 text-sm text-destructive"
        >
          <X className="mt-0.5 h-4 w-4 shrink-0" strokeWidth={1.5} aria-hidden />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
