import { X } from "lucide-react";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getFormat(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot >= 0 ? filename.slice(dot + 1).toUpperCase() : "—";
}

interface FileInfoCardProps {
  file: File;
  onRemove: () => void;
}

export function FileInfoCard({ file, onRemove }: FileInfoCardProps) {
  const format = getFormat(file.name);

  return (
    <div className="border border-border bg-background">
      {/* Card header */}
      <div className="flex items-center justify-between border-b border-border px-6 py-4">
        <span className="label-mono">File selected</span>
        <span className="font-mono text-[0.65rem] tracking-[0.14em] text-signal">{format}</span>
      </div>

      {/* Filename */}
      <div className="px-6 py-5">
        <p
          className="truncate text-[0.95rem] text-foreground"
          title={file.name}
        >
          {file.name}
        </p>

        {/* Metrics row */}
        <dl className="mt-6 grid grid-cols-2 gap-px border border-border bg-border sm:grid-cols-2">
          <div className="bg-background px-4 py-4">
            <dt className="label-mono">Format</dt>
            <dd className="mt-2 font-mono text-[0.85rem] text-foreground">{format}</dd>
          </div>
          <div className="bg-background px-4 py-4">
            <dt className="label-mono">Size</dt>
            <dd className="mt-2 font-mono text-[0.85rem] text-foreground">{formatBytes(file.size)}</dd>
          </div>
        </dl>

        {/* Remove */}
        <div className="mt-6 flex justify-end">
          <button
            type="button"
            onClick={onRemove}
            aria-label={`Remove selected file: ${file.name}`}
            className="inline-flex items-center gap-2 border border-border px-4 py-2 text-[0.8rem] text-muted-foreground transition-colors duration-200 hover:border-border-strong hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          >
            <X className="h-3.5 w-3.5" strokeWidth={1.5} aria-hidden />
            Remove
          </button>
        </div>
      </div>
    </div>
  );
}
