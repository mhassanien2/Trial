"use client";

import { useCallback, useEffect, useState } from "react";
import { Gavel } from "lucide-react";
import { useTranslations } from "next-intl";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface Persona {
  key: string;
  critique: string;
  strengths: string[];
  concerns: string[];
  questions: Array<{ question: string; suggestedAnswer: string }>;
  aiDisabled?: boolean;
}

export function MockPanel({
  programId,
  canRun,
  ssrDocs,
  latestRunId,
}: {
  programId: string;
  canRun: boolean;
  ssrDocs: Array<{ id: string; title: string }>;
  latestRunId: string | null;
}) {
  const t = useTranslations("panel");
  const [docId, setDocId] = useState(ssrDocs[0]?.id ?? "");
  const [runId, setRunId] = useState<string | null>(latestRunId);
  const [status, setStatus] = useState<string>("");
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [running, setRunning] = useState(false);

  const poll = useCallback(async (id: string) => {
    const res = await fetch(`/api/panel/${id}`);
    if (!res.ok) return;
    const { run } = (await res.json()) as {
      run: { status: string; resultsJson: { personas?: Persona[] } };
    };
    setStatus(run.status);
    setPersonas(run.resultsJson?.personas ?? []);
    if (["PENDING", "RUNNING"].includes(run.status)) {
      void fetch("/api/jobs/run", { method: "POST" });
    }
  }, []);

  useEffect(() => {
    if (!runId) return;
    // setState happens only after the fetch resolves, not synchronously.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void poll(runId);
  }, [runId, poll]);

  useEffect(() => {
    if (!runId || !["PENDING", "RUNNING"].includes(status)) return;
    const iv = setInterval(() => void poll(runId), 3000);
    return () => clearInterval(iv);
  }, [runId, status, poll]);

  async function run() {
    if (!docId) return;
    setRunning(true);
    const res = await fetch("/api/panel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ documentId: docId, programId }),
    });
    setRunning(false);
    if (res.ok) {
      const { id } = (await res.json()) as { id: string };
      setPersonas([]);
      setStatus("PENDING");
      setRunId(id);
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Gavel className="h-5 w-5" />
              {t("title")}
            </CardTitle>
            <p className="text-sm text-muted-foreground">{t("subtitle")}</p>
          </div>
          {canRun ? (
            ssrDocs.length === 0 ? (
              <span className="text-sm text-muted-foreground">{t("noDocs")}</span>
            ) : (
              <div className="flex items-center gap-2">
                <select
                  value={docId}
                  onChange={(e) => setDocId(e.target.value)}
                  className="h-9 rounded-md border border-input bg-transparent px-3 text-sm shadow-sm"
                >
                  {ssrDocs.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.title}
                    </option>
                  ))}
                </select>
                <Button onClick={run} disabled={running || !docId}>
                  {running ? t("running") : t("run")}
                </Button>
              </div>
            )
          ) : null}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {["PENDING", "RUNNING"].includes(status) ? (
          <p className="text-sm text-muted-foreground">{t("processing")}</p>
        ) : null}

        {personas.length > 0 ? (
          <div className="grid gap-4 lg:grid-cols-3">
            {personas.map((p) => (
              <div key={p.key} className="space-y-3 rounded-lg border p-4">
                <h3 className="font-semibold">{t(`personas.${p.key}`)}</h3>
                {p.aiDisabled ? (
                  <p className="text-sm text-muted-foreground">{t("aiDisabled")}</p>
                ) : (
                  <>
                    <p className="text-sm">{p.critique}</p>
                    {p.strengths.length > 0 ? (
                      <div>
                        <p className="text-xs font-semibold uppercase text-muted-foreground">
                          {t("strengths")}
                        </p>
                        <ul className="ms-4 list-disc text-sm">
                          {p.strengths.map((s, i) => (
                            <li key={i}>{s}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {p.concerns.length > 0 ? (
                      <div>
                        <p className="text-xs font-semibold uppercase text-muted-foreground">
                          {t("concerns")}
                        </p>
                        <ul className="ms-4 list-disc text-sm">
                          {p.concerns.map((s, i) => (
                            <li key={i}>{s}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {p.questions.length > 0 ? (
                      <div className="space-y-2">
                        <p className="text-xs font-semibold uppercase text-muted-foreground">
                          {t("questions")}
                        </p>
                        {p.questions.map((q, i) => (
                          <div key={i} className="rounded-md bg-muted p-2 text-sm">
                            <p className="font-medium">{q.question}</p>
                            <p className="mt-1 text-muted-foreground">
                              <Badge variant="outline" className="me-1">
                                {t("suggestedAnswer")}
                              </Badge>
                              {q.suggestedAnswer}
                            </p>
                          </div>
                        ))}
                      </div>
                    ) : null}
                  </>
                )}
              </div>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
