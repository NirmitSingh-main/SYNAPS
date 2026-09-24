import { useState } from "react";

export interface ActivityPoint {
  date: string; // formatted label e.g., "Sep 20" or "09/20"
  fullDate?: string;
  count: number;
}

interface ActivityLineChartProps {
  data: ActivityPoint[];
  title?: string;
  subtitle?: string;
}

export function ActivityLineChart({
  data,
  title = "Analysis Activity",
  subtitle = "Your recent analysis frequency",
}: ActivityLineChartProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const width = 500;
  const height = 180;
  const padLeft = 36;
  const padRight = 20;
  const padTop = 20;
  const padBottom = 30;

  const chartWidth = width - padLeft - padRight;
  const chartHeight = height - padTop - padBottom;

  const maxCount = Math.max(...data.map((d) => d.count), 1);
  // Nice y-max step
  const yMax = Math.max(Math.ceil(maxCount * 1.2), 4);

  const points = data.map((d, i) => {
    const x =
      data.length > 1
        ? padLeft + (i / (data.length - 1)) * chartWidth
        : padLeft + chartWidth / 2;
    const y = padTop + chartHeight - (d.count / yMax) * chartHeight;
    return { x, y, ...d };
  });

  // SVG path string
  const linePath =
    points.length > 0
      ? points.reduce(
          (acc, p, i) => `${acc} ${i === 0 ? "M" : "L"} ${p.x},${p.y}`,
          ""
        )
      : "";

  const areaPath =
    points.length > 0 && points[0] && points[points.length - 1]
      ? `${linePath} L ${points[points.length - 1]!.x},${padTop + chartHeight} L ${
          points[0]!.x
        },${padTop + chartHeight} Z`
      : "";

  const totalActivity = data.reduce((acc, d) => acc + d.count, 0);

  return (
    <div className="flex h-full flex-col justify-between border border-border bg-surface/30 p-5">
      <div>
        <div className="flex items-center justify-between">
          <p className="label-mono">{title}</p>
          <span className="font-mono text-[0.7rem] text-signal">
            {totalActivity} total in period
          </span>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>
      </div>

      {data.length === 0 || totalActivity === 0 ? (
        <div className="my-8 flex flex-col items-center justify-center py-6 text-center">
          <div className="h-20 w-48 border border-dashed border-border/80 flex items-center justify-center">
            <span className="font-mono text-xs text-muted-foreground">NO ACTIVITY</span>
          </div>
          <p className="mt-3 font-mono text-xs text-muted-foreground">
            No analysis events recorded yet
          </p>
        </div>
      ) : (
        <div className="relative my-2 w-full">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full h-auto overflow-visible select-none"
          >
            <defs>
              <linearGradient id="activityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--signal)" stopOpacity="0.28" />
                <stop offset="100%" stopColor="var(--signal)" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Grid lines (horizontal) */}
            {[0, 0.33, 0.66, 1].map((ratio, idx) => {
              const y = padTop + chartHeight * (1 - ratio);
              const val = Math.round(yMax * ratio);
              return (
                <g key={idx}>
                  <line
                    x1={padLeft}
                    y1={y}
                    x2={padLeft + chartWidth}
                    y2={y}
                    stroke="var(--border)"
                    strokeDasharray="2 2"
                    strokeWidth="1"
                  />
                  <text
                    x={padLeft - 6}
                    y={y + 3}
                    textAnchor="end"
                    className="fill-muted-foreground font-mono text-[9px]"
                  >
                    {val}
                  </text>
                </g>
              );
            })}

            {/* Area fill */}
            <path d={areaPath} fill="url(#activityGradient)" />

            {/* Line trace */}
            <path
              d={linePath}
              fill="none"
              stroke="var(--signal)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Points and hover targets */}
            {points.map((p, i) => {
              const isHovered = hoveredIndex === i;
              return (
                <g key={i}>
                  {/* Invisible larger hover area */}
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={14}
                    fill="transparent"
                    className="cursor-pointer"
                    onMouseEnter={() => setHoveredIndex(i)}
                    onMouseLeave={() => setHoveredIndex(null)}
                  />
                  {/* Visual point */}
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={isHovered ? 5 : 3}
                    fill="var(--background)"
                    stroke="var(--signal)"
                    strokeWidth={isHovered ? 2.5 : 1.5}
                    className="pointer-events-none transition-all duration-150"
                  />
                </g>
              );
            })}

            {/* X-axis date labels (sampled for clarity) */}
            {points
              .filter((_, idx) => {
                if (points.length <= 6) return true;
                const step = Math.ceil(points.length / 5);
                return idx % step === 0 || idx === points.length - 1;
              })
              .map((p, idx) => (
                <text
                  key={idx}
                  x={p.x}
                  y={height - 8}
                  textAnchor="middle"
                  className="fill-muted-foreground font-mono text-[9px]"
                >
                  {p.date}
                </text>
              ))}
          </svg>

          {/* Floating Tooltip */}
          {hoveredIndex !== null && points[hoveredIndex] && (
            <div
              className="pointer-events-none absolute -top-2 transform -translate-x-1/2 -translate-y-full rounded border border-border-strong bg-background/95 px-2.5 py-1.5 shadow-md backdrop-blur-sm"
              style={{
                left: `${(points[hoveredIndex].x / width) * 100}%`,
              }}
            >
              <div className="font-mono text-[0.65rem] text-muted-foreground">
                {points[hoveredIndex].fullDate || points[hoveredIndex].date}
              </div>
              <div className="font-mono text-xs font-semibold text-signal">
                {points[hoveredIndex].count}{" "}
                {points[hoveredIndex].count === 1 ? "analysis" : "analyses"}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
