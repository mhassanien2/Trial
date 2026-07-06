"use client";

import { useCallback, useEffect, useState } from "react";
import { useTranslations } from "next-intl";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Link, useRouter } from "@/i18n/navigation";

interface ReviewRow {
  id: string;
  status: string;
  readinessScore: number | null;
  createdAt: string;
  document: { title: string };
  pack: { nameEn: string; code: string };
  program: { code: string } | null;
}

const STATUS_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  COMPLETED: "default",
  RUNNING: "secondary",
  PENDING: "secondary",
  FAILED: "destructive",
};

export function ReviewsView({
  canRun,
  documents,
  packs,
  programs,
}: {
  canRun: boolean;
  documents: Array<{ id: string; title: string }>;
  packs: Array<{ id: string; name: string; code: string }>;
  programs: Array<{ id: string; label: string }>;
}) {
  const t = useTranslations("reviews");
  const router = useRouter();

  const [reviews, setReviews] = useState<ReviewRow[]>([]);
  const [documentId, setDocumentId] = useState(documents[0]?.id ?? "");
  const [packId, setPackId] = useState(packs[0]?.id ?? "");
  const [programId, setProgramId] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const res = await fetch("/api/reviews");
    if (res.ok) {
      const json = (await res.json()) as { reviews: ReviewRow[] };
      setReviews(json.reviews);
      if (json.reviews.some((r) => ["PENDING", "RUNNING"].includes(r.status))) {
        void fetch("/api/jobs/run", { method: "POST" });
      }
    }
  }, []);

  useEffect(() => {
    // setState happens only after the fetch resolves, not synchronously.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
    const id = setInterval(() => void refresh(), 3500);
    return () => clearInterval(id);
  }, [refresh]);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    setRunning(true);
    setError(null);
    const res = await fetch("/api/reviews", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        documentId,
        packId,
        programId: programId || undefined,
      }),
    });
    if (!res.ok) {
      const json = (await res.json().catch(() => null)) as { error?: string } | null;
      setError(json?.error ?? "Failed");
      setRunning(false);
      return;
    }
    const { id } = (await res.json()) as { id: string };
    router.push(`/reviews/${id}`);
  }

  return (
    <div className="space-y-6">
      {canRun ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t("newReview")}</CardTitle>
          </CardHeader>
          <CardContent>
            {documents.length === 0 ? (
              <p className="text-sm text-muted-foreground">{t("noReadyDocs")}</p>
            ) : (
              <form onSubmit={run} className="grid gap-4 sm:grid-cols-3">
                <div className="space-y-2">
                  <Label htmlFor="doc">{t("document")}</Label>
                  <select
                    id="doc"
                    value={documentId}
                    onChange={(e) => setDocumentId(e.target.value)}
                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
                  >
                    {documents.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.title}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="pack">{t("pack")}</Label>
                  <select
                    id="pack"
                    value={packId}
                    onChange={(e) => setPackId(e.target.value)}
                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
                  >
                    {packs.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.code} — {p.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="prog">{t("program")}</Label>
                  <select
                    id="prog"
                    value={programId}
                    onChange={(e) => setProgramId(e.target.value)}
                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
                  >
                    <option value="">{t("noProgram")}</option>
                    {programs.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.label}
                      </option>
                    ))}
                  </select>
                </div>
                {error ? (
                  <p role="alert" className="text-sm text-destructive sm:col-span-3">
                    {error}
                  </p>
                ) : null}
                <div className="sm:col-span-3">
                  <Button type="submit" disabled={running || !documentId || !packId}>
                    {running ? t("running") : t("run")}
                  </Button>
                </div>
              </form>
            )}
          </CardContent>
        </Card>
      ) : null}

      <Card>
        <CardContent className="p-0">
          {reviews.length === 0 ? (
            <p className="p-6 text-sm text-muted-foreground">{t("empty")}</p>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="p-3 text-start font-medium">{t("document")}</th>
                  <th className="p-3 text-start font-medium">{t("pack")}</th>
                  <th className="p-3 text-start font-medium">{t("status")}</th>
                  <th className="p-3 text-start font-medium">{t("score")}</th>
                  <th className="p-3 text-start font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {reviews.map((r) => (
                  <tr key={r.id} className="border-b last:border-0">
                    <td className="p-3 font-medium">{r.document.title}</td>
                    <td className="p-3">{r.pack.code}</td>
                    <td className="p-3">
                      <Badge variant={STATUS_VARIANT[r.status] ?? "outline"}>
                        {t(`statuses.${r.status}`)}
                      </Badge>
                    </td>
                    <td className="p-3 font-semibold">
                      {r.readinessScore != null ? `${r.readinessScore}/100` : "—"}
                    </td>
                    <td className="p-3">
                      <Link
                        href={`/reviews/${r.id}`}
                        className="text-primary hover:underline"
                      >
                        {t("view")}
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
