import { useState } from "react";

export interface ModulationSlice {
  label: string;
  count: number;
  color: string;
}

interface ModulationDonutChartProps {
  data: { label: string; count: number }[];
  title?: string;
  subtitle?: string;
  totalLabel?: string;
}

const COLOR_MAP: Record<string, string> = {
  BPSK: "#38bdf8", // Sky blue
  QPSK: "#818cf8", // Indigo
  FSK: "#34d399",  // Emerald
  QAM16: "#f472b6", // Pink
  QAM64: "#fb923c", // Orange
  AM: "#a78bfa",    // Violet
  FM: "#fbbf24",    // Amber
};

const DEFAULT_COLORS = [
  "#38bdf8",
  "#818cf8",
  "#34d399",
  "#f472b6",
  "#fb923c",
  "#a78bfa",
];

export function ModulationDonutChart({
  data,
  title = "Modulation Distribution",
  subtitle = "Classification breakdown across your analyses",
  totalLabel = "TOTAL",
}: ModulationDonutChartProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const total = data.reduce((acc, d) => acc + d.count, 0);

  const slices: (ModulationSlice & {
    percentage: number;
    startAngle: number;
    endAngle: number;
  })[] = [];

  let accumulatedAngle = -90; // Start at top

  data.forEach((item, index) => {
    const percentage = total > 0 ? (item.count / total) * 100 : 0;
    const angleSpan = total > 0 ? (item.count / total) * 360 : 0;
    const startAngle = accumulatedAngle;
    const endAngle = accumulatedAngle + angleSpan;
    accumulatedAngle += angleSpan;

    const color =
      COLOR_MAP[item.label.toUpperCase()] ||
      DEFAULT_COLORS[index % DEFAULT_COLORS.length] ||
      "#38bdf8";

    slices.push({
      label: item.label,
      count: item.count,
      percentage,
      color,
      startAngle,
      endAngle,
    });
  });

  const size = 200;
  const center = size / 2;
  const radius = 75;
  const innerRadius = 52;

  function polarToCartesian(
    centerX: number,
    centerY: number,
    r: number,
    angleInDegrees: number
  ) {
    const angleInRadians = ((angleInDegrees - 90) * Math.PI) / 180.0;
    return {
      x: centerX + r * Math.cos(angleInRadians),
      y: centerY + r * Math.sin(angleInRadians),
    };
  }

  function describeArc(
    x: number,
    y: number,
    r: number,
    innerR: number,
    startAngle: number,
    endAngle: number
  ) {
    // If it's a full circle or close to it
    const sweep = endAngle - startAngle;
    const clampedEndAngle = sweep >= 359.99 ? startAngle + 359.99 : endAngle;

    const start = polarToCartesian(x, y, r, clampedEndAngle);
    const end = polarToCartesian(x, y, r, startAngle);
    const innerStart = polarToCartesian(x, y, innerR, startAngle);
    const innerEnd = polarToCartesian(x, y, innerR, clampedEndAngle);

    const largeArcFlag = clampedEndAngle - startAngle <= 180 ? "0" : "1";

    return [
      "M", start.x, start.y,
      "A", r, r, 0, largeArcFlag, 0, end.x, end.y,
      "L", innerStart.x, innerStart.y,
      "A", innerR, innerR, 0, largeArcFlag, 1, innerEnd.x, innerEnd.y,
      "Z",
    ].join(" ");
  }

  return (
    <div className="flex h-full flex-col justify-between border border-border bg-surface/30 p-5">
      <div>
        <div className="flex items-center justify-between">
          <p className="label-mono">{title}</p>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>
      </div>

      {total === 0 ? (
        <div className="my-8 flex flex-col items-center justify-center py-6 text-center">
          <div className="h-28 w-28 rounded-full border border-dashed border-border/80 flex items-center justify-center">
            <span className="font-mono text-xs text-muted-foreground">0 ANALYSES</span>
          </div>
          <p className="mt-3 font-mono text-xs text-muted-foreground">
            No modulation data yet
          </p>
        </div>
      ) : (
        <div className="my-4 flex flex-col items-center gap-6 sm:flex-row sm:justify-around">
          {/* Donut graphic */}
          <div className="relative shrink-0">
            <svg
              width={size}
              height={size}
              viewBox={`0 0 ${size} ${size}`}
              className="overflow-visible"
            >
              {slices.map((slice, i) => {
                if (slice.count === 0) return null;
                const isHovered = hoveredIndex === i;
                const pathData = describeArc(
                  center,
                  center,
                  isHovered ? radius + 4 : radius,
                  innerRadius,
                  slice.startAngle + 90,
                  slice.endAngle + 90
                );

                return (
                  <path
                    key={slice.label}
                    d={pathData}
                    fill={slice.color}
                    opacity={hoveredIndex === null || isHovered ? 0.9 : 0.4}
                    className="cursor-pointer transition-all duration-200"
                    onMouseEnter={() => setHoveredIndex(i)}
                    onMouseLeave={() => setHoveredIndex(null)}
                  />
                );
              })}
            </svg>

            {/* Center Label */}
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-center">
              {hoveredIndex !== null && slices[hoveredIndex] ? (
                <>
                  <span className="font-mono text-[0.65rem] tracking-wider text-muted-foreground uppercase">
                    {slices[hoveredIndex].label}
                  </span>
                  <span className="font-mono text-lg font-semibold text-foreground">
                    {slices[hoveredIndex].percentage.toFixed(1)}%
                  </span>
                  <span className="font-mono text-[0.65rem] text-muted-foreground">
                    {slices[hoveredIndex].count} signals
                  </span>
                </>
              ) : (
                <>
                  <span className="font-mono text-[0.65rem] tracking-widest text-muted-foreground">
                    {totalLabel}
                  </span>
                  <span className="font-mono text-xl font-bold text-foreground">
                    {total}
                  </span>
                </>
              )}
            </div>
          </div>

          {/* Legend */}
          <div className="w-full max-w-xs space-y-2">
            {slices.map((slice, i) => (
              <div
                key={slice.label}
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
                className={`flex cursor-pointer items-center justify-between rounded-sm border px-2.5 py-1.5 transition-all ${
                  hoveredIndex === i
                    ? "border-border-strong bg-surface/80"
                    : "border-transparent hover:bg-surface/50"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span
                    className="h-2.5 w-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: slice.color }}
                  />
                  <span className="font-mono text-xs text-foreground">
                    {slice.label}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xs font-medium text-foreground">
                    {slice.count}
                  </span>
                  <span className="w-11 text-right font-mono text-[0.7rem] text-muted-foreground">
                    {slice.percentage.toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
