"use client";

import { useCallback, useEffect, useState } from "react";
import { Download } from "lucide-react";
import { useTranslations } from "next-intl";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface Finding {
  id: string;
  verdict: "MET" | "PARTIALLY_MET" | "NOT_MET" | "NOT_EVALUATED";
  score: number | null;
  findingText: string;
  citations: Array<{ page: number | null; quote: string }>;
  recommendations: string[];
  criterion: {
    code: string;
    titleEn: string;
    standard: { code: string; titleEn: string; sortOrder: number };
  };
}

interface ReviewData {
  id: string;
  status: string;
  readinessScore: number | null;
  summary: string | null;
  error: string | null;
  findings: Finding[];
}

const VERDICT_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  MET: "default",
  PARTIALLY_MET: "secondary",
  NOT_MET: "destructive",
  NOT_EVALUATED: "outline",
};

function isPlaceholder(s: string) {
  return !s || s.includes("TODO:OFFICIAL_TEXT");
}

export function ReviewDetail({
  reviewId,
  documentTitle,
  packName,
  programLabel,
  initialStatus,
}: {
  reviewId: string;
  documentTitle: string;
  packName: string;
  programLabel: string | null;
  initialStatus: string;
}) {
  const t = useTranslations("reviews");
  const [data, setData] = useState<ReviewData | null>(null);
  const [status, setStatus] = useState(initialStatus);

  const refresh = useCallback(async () => {
    const res = await fetch(`/api/reviews/${reviewId}`);
    if (res.ok) {
      const json = (await res.json()) as { review: ReviewData };
      setData(json.review);
      setStatus(json.review.status);
      if (["PENDING", "RUNNING"].includes(json.review.status)) {
        void fetch("/api/jobs/run", { method: "POST" });
      }
    }
  }, [reviewId]);

  useEffect(() => {
    // setState happens only after the fetch resolves, not synchronously.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!["PENDING", "RUNNING"].includes(status)) return;
    const id = setInterval(() => void refresh(), 3000);
    return () => clearInterval(id);
  }, [status, refresh]);

  const scoreColor = (score: number) =>
    score >= 75 ? "text-green-600" : score >= 50 ? "text-amber-600" : "text-red-600";

  // Group findings by standard, ordered.
  const groups = new Map<string, { title: string; sortOrder: number; findings: Finding[] }>();
  for (const f of data?.findings ?? []) {
    const key = f.criterion.standard.code;
    if (!groups.has(key)) {
      groups.set(key, {
        title: isPlaceholder(f.criterion.standard.titleEn)
          ? f.criterion.standard.code
          : `${f.criterion.standard.code} — ${f.criterion.standard.titleEn}`,
        sortOrder: f.criterion.standard.sortOrder,
        findings: [],
      });
    }
    groups.get(key)!.findings.push(f);
  }
  const orderedGroups = [...groups.values()].sort((a, b) => a.sortOrder - b.sortOrder);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">{documentTitle}</h1>
          <p className="text-sm text-muted-foreground">
            {packName}
            {programLabel ? ` · ${programLabel}` : ""}
          </p>
        </div>
        {status === "COMPLETED" ? (
          <Button asChild variant="outline">
            <a href={`/api/reviews/${reviewId}/report`}>
              <Download />
              {t("exportReport")}
            </a>
          </Button>
        ) : null}
      </div>

      {status === "PENDING" || status === "RUNNING" ? (
        <Card>
          <CardContent className="p-6 text-sm text-muted-foreground">
            {t("processing")}
          </CardContent>
        </Card>
      ) : null}

      {status === "FAILED" ? (
        <Card>
          <CardContent className="p-6 text-sm text-destructive">
            {t("failed")}: {data?.error}
          </CardContent>
        </Card>
      ) : null}

      {status === "COMPLETED" && data ? (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">{t("readinessScore")}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-baseline gap-2">
                <span className={`text-5xl font-bold ${scoreColor(data.readinessScore ?? 0)}`}>
                  {data.readinessScore ?? 0}
                </span>
                <span className="text-muted-foreground">{t("outOf")}</span>
              </div>
              {data.summary ? (
                <p className="mt-2 text-sm text-muted-foreground">{data.summary}</p>
              ) : null}
            </CardContent>
          </Card>

          <div className="space-y-4">
            <h2 className="text-lg font-semibold">{t("findings")}</h2>
            {orderedGroups.map((group) => (
              <Card key={group.title}>
                <CardHeader>
                  <CardTitle className="text-base">{group.title}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {group.findings.map((f) => (
                    <div key={f.id} className="rounded-md border p-3">
                      <div className="mb-2 flex flex-wrap items-center gap-2">
                        <Badge variant={VERDICT_VARIANT[f.verdict]}>
                          {t(`verdicts.${f.verdict}`)}
                        </Badge>
                        <span className="text-sm font-medium">
                          {f.criterion.code}
                          {!isPlaceholder(f.criterion.titleEn)
                            ? ` — ${f.criterion.titleEn}`
                            : ""}
                        </span>
                        {f.score != null ? (
                          <span className="text-xs text-muted-foreground">{f.score}/100</span>
                        ) : null}
                      </div>
                      <p className="text-sm">{f.findingText}</p>

                      {f.citations.length > 0 ? (
                        <div className="mt-2">
                          <p className="text-xs font-semibold uppercase text-muted-foreground">
                            {t("citations")}
                          </p>
                          <ul className="mt-1 space-y-1">
                            {f.citations.map((c, i) => (
                              <li key={i} className="border-s-2 ps-2 text-xs text-muted-foreground">
                                {c.page != null ? `p.${c.page}: ` : ""}“{c.quote}”
                              </li>
                            ))}
                          </ul>
                        </div>
                      ) : (
                        <p className="mt-2 text-xs italic text-muted-foreground">
                          {t("noCitations")}
                        </p>
                      )}

                      {f.recommendations.length > 0 ? (
                        <div className="mt-2">
                          <p className="text-xs font-semibold uppercase text-muted-foreground">
                            {t("recommendations")}
                          </p>
                          <ul className="ms-4 list-disc text-sm">
                            {f.recommendations.map((r, i) => (
                              <li key={i}>{r}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                    </div>
                  ))}
                </CardContent>
              </Card>
            ))}
          </div>
        </>
      ) : null}
    </div>
  );
}
