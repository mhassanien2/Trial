"use client";

import { useState } from "react";
import { GitCompareArrows } from "lucide-react";
import { useLocale, useTranslations } from "next-intl";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";

interface Pair {
  from: { standardCode: string; code: string; title: string };
  to: { standardCode: string; code: string; title: string };
  overlapScore: number;
  fromHasEvidence: boolean;
  toHasEvidence: boolean;
}
interface Result {
  fromPack: string;
  toPack: string;
  pairs: Pair[];
  fromCovered: number;
  toCovered: number;
  bothCovered: number;
}

export function CrossMapper({
  programId,
  packs,
}: {
  programId: string;
  packs: Array<{ id: string; label: string }>;
}) {
  const t = useTranslations("mapper");
  const locale = useLocale();
  const [fromPackId, setFromPackId] = useState(packs[0]?.id ?? "");
  const [toPackId, setToPackId] = useState(packs[1]?.id ?? "");
  const [result, setResult] = useState<Result | null>(null);
  const [computing, setComputing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function compute() {
    if (fromPackId === toPackId) {
      setError(t("samePackError"));
      return;
    }
    setError(null);
    setComputing(true);
    const res = await fetch("/api/mapper", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ programId, fromPackId, toPackId, locale }),
    });
    setComputing(false);
    if (res.ok) {
      const { result } = (await res.json()) as { result: Result };
      setResult(result);
    } else {
      const j = (await res.json().catch(() => null)) as { error?: string } | null;
      setError(j?.error ?? "Failed");
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <GitCompareArrows className="h-5 w-5" />
          {t("title")}
        </CardTitle>
        <p className="text-sm text-muted-foreground">{t("subtitle")}</p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="space-y-1">
            <Label htmlFor="from">{t("fromPack")}</Label>
            <select
              id="from"
              value={fromPackId}
              onChange={(e) => setFromPackId(e.target.value)}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm shadow-sm"
            >
              {packs.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <Label htmlFor="to">{t("toPack")}</Label>
            <select
              id="to"
              value={toPackId}
              onChange={(e) => setToPackId(e.target.value)}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm shadow-sm"
            >
              {packs.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>
          <div className="flex items-end">
            <Button onClick={compute} disabled={computing}>
              {computing ? t("computing") : t("compute")}
            </Button>
          </div>
        </div>
        {error ? <p className="text-sm text-destructive">{error}</p> : null}

        {result ? (
          <div className="space-y-3">
            <div className="flex flex-wrap gap-4 text-sm">
              <Badge variant="outline">
                {t("fromEvidence")}: {result.fromCovered}
              </Badge>
              <Badge variant="outline">
                {t("toEvidence")}: {result.toCovered}
              </Badge>
              <Badge>
                {t("both")}: {result.bothCovered}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">{t("note")}</p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-muted-foreground">
                    <th className="p-2 text-start font-medium">
                      {result.fromPack} — {t("primaryCriterion")}
                    </th>
                    <th className="p-2 text-start font-medium">
                      {result.toPack} — {t("mappedCriterion")}
                    </th>
                    <th className="p-2 text-start font-medium">{t("overlapScore")}</th>
                    <th className="p-2 text-start font-medium">{t("fromEvidence")}</th>
                    <th className="p-2 text-start font-medium">{t("toEvidence")}</th>
                  </tr>
                </thead>
                <tbody>
                  {result.pairs.map((p, i) => (
                    <tr key={i} className="border-b last:border-0">
                      <td className="p-2">
                        <span className="me-1 text-muted-foreground">
                          {p.from.standardCode}·{p.from.code}
                        </span>
                        {p.from.title}
                      </td>
                      <td className="p-2">
                        <span className="me-1 text-muted-foreground">
                          {p.to.standardCode}·{p.to.code}
                        </span>
                        {p.to.title}
                      </td>
                      <td className="p-2">
                        <div className="flex items-center gap-2">
                          <div className="h-2 w-16 overflow-hidden rounded bg-muted">
                            <div
                              className="h-full bg-primary"
                              style={{ width: `${Math.round(p.overlapScore * 100)}%` }}
                            />
                          </div>
                          <span className="text-xs text-muted-foreground">
                            {(p.overlapScore * 100).toFixed(0)}%
                          </span>
                        </div>
                      </td>
                      <td className="p-2">
                        {p.fromHasEvidence ? (
                          <span className="text-primary">{t("yes")}</span>
                        ) : (
                          <span className="text-muted-foreground">{t("no")}</span>
                        )}
                      </td>
                      <td className="p-2">
                        {p.toHasEvidence ? (
                          <span className="text-primary">{t("yes")}</span>
                        ) : (
                          <span className="text-muted-foreground">{t("no")}</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
