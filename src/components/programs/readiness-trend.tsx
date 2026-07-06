import { getTranslations } from "next-intl/server";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface Point {
  score: number;
  createdAt: string; // ISO
}

/**
 * Inline SVG sparkline of readiness over time — no chart dependency.
 * Server component: renders directly from ReadinessSnapshot rows.
 */
export async function ReadinessTrend({ points }: { points: Point[] }) {
  const t = await getTranslations("trend");

  if (points.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">{t("title")}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{t("empty")}</p>
        </CardContent>
      </Card>
    );
  }

  const W = 640;
  const H = 160;
  const PAD = 28;
  const n = points.length;
  const xFor = (i: number) => (n === 1 ? W / 2 : PAD + (i * (W - 2 * PAD)) / (n - 1));
  const yFor = (score: number) => H - PAD - (score / 100) * (H - 2 * PAD);

  const linePath = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xFor(i).toFixed(1)} ${yFor(p.score).toFixed(1)}`)
    .join(" ");
  const areaPath = `${linePath} L ${xFor(n - 1).toFixed(1)} ${H - PAD} L ${xFor(0).toFixed(1)} ${H - PAD} Z`;

  const latest = points[n - 1].score;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-lg">{t("title")}</CardTitle>
          <p className="text-sm text-muted-foreground">{t("subtitle")}</p>
        </div>
        <div className="text-end">
          <div className="text-2xl font-bold">{latest}/100</div>
          <div className="text-xs text-muted-foreground">{t("latest")}</div>
        </div>
      </CardHeader>
      <CardContent>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full"
          role="img"
          aria-label={t("subtitle")}
          preserveAspectRatio="xMidYMid meet"
        >
          {/* gridlines at 0/50/100 */}
          {[0, 50, 100].map((g) => (
            <g key={g}>
              <line
                x1={PAD}
                x2={W - PAD}
                y1={yFor(g)}
                y2={yFor(g)}
                className="stroke-border"
                strokeWidth={1}
              />
              <text
                x={4}
                y={yFor(g) + 4}
                className="fill-muted-foreground"
                fontSize={10}
              >
                {g}
              </text>
            </g>
          ))}
          <path d={areaPath} className="fill-primary/10" />
          <path d={linePath} className="stroke-primary" strokeWidth={2} fill="none" />
          {points.map((p, i) => (
            <circle
              key={i}
              cx={xFor(i)}
              cy={yFor(p.score)}
              r={3}
              className="fill-primary"
            >
              <title>
                {new Date(p.createdAt).toLocaleDateString()}: {p.score}/100
              </title>
            </circle>
          ))}
        </svg>
        <p className="mt-1 text-xs text-muted-foreground">
          {t("points", { count: n })}
        </p>
      </CardContent>
    </Card>
  );
}
