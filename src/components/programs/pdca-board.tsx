"use client";

import { useState } from "react";
import { Plus } from "lucide-react";
import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Link } from "@/i18n/navigation";

type Phase = "PLAN" | "DO" | "CHECK" | "ACT";
const PHASES: Phase[] = ["PLAN", "DO", "CHECK", "ACT"];

const NEXT_STATUS: Record<string, string> = {
  PLANNED: "IN_PROGRESS",
  IN_PROGRESS: "COMPLETED",
  COMPLETED: "PLANNED",
  BLOCKED: "IN_PROGRESS",
};

const STATUS_VARIANT: Record<string, "default" | "secondary" | "outline" | "destructive"> = {
  COMPLETED: "default",
  IN_PROGRESS: "secondary",
  PLANNED: "outline",
  BLOCKED: "destructive",
};

interface PlanItem {
  id: string;
  phase: Phase;
  title: string;
  status: string;
}

export function PdcaBoard({
  programId,
  canManage,
  planItems,
  drafts,
  reviews,
  actionsCount,
}: {
  programId: string;
  canManage: boolean;
  planItems: PlanItem[];
  drafts: Array<{ id: string; title: string; status: string }>;
  reviews: Array<{ id: string; status: string; score: number | null; packCode: string }>;
  actionsCount: number;
}) {
  const t = useTranslations("pdca");
  const router = useRouter();
  const [adding, setAdding] = useState<Phase | null>(null);
  const [title, setTitle] = useState("");

  async function addItem(phase: Phase) {
    if (!title.trim()) return;
    await fetch(`/api/programs/${programId}/plan-items`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phase, title }),
    });
    setTitle("");
    setAdding(null);
    router.refresh();
  }

  async function advance(item: PlanItem) {
    await fetch(`/api/programs/${programId}/plan-items`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ itemId: item.id, status: NEXT_STATUS[item.status] }),
    });
    router.refresh();
  }

  const hint: Record<Phase, string> = {
    PLAN: t("planHint"),
    DO: t("doHint"),
    CHECK: t("checkHint"),
    ACT: t("actHint"),
  };

  return (
    <div>
      <h2 className="mb-3 text-lg font-semibold">{t("board")}</h2>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {PHASES.map((phase) => {
          const items = planItems.filter((p) => p.phase === phase);
          return (
            <Card key={phase} className="flex flex-col">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center justify-between text-base">
                  {t(phase)}
                  {canManage ? (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => setAdding(adding === phase ? null : phase)}
                      aria-label={t("addItem")}
                    >
                      <Plus className="h-4 w-4" />
                    </Button>
                  ) : null}
                </CardTitle>
                <p className="text-xs text-muted-foreground">{hint[phase]}</p>
              </CardHeader>
              <CardContent className="flex-1 space-y-2">
                {adding === phase ? (
                  <div className="space-y-2 rounded-md border p-2">
                    <Input
                      autoFocus
                      value={title}
                      onChange={(e) => setTitle(e.target.value)}
                      placeholder={t("newItemTitle")}
                      onKeyDown={(e) => e.key === "Enter" && addItem(phase)}
                    />
                    <div className="flex gap-2">
                      <Button size="sm" onClick={() => addItem(phase)}>
                        {t("add")}
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => setAdding(null)}>
                        {t("cancel")}
                      </Button>
                    </div>
                  </div>
                ) : null}

                {items.map((item) => (
                  <div key={item.id} className="rounded-md border p-2 text-sm">
                    <p>{item.title}</p>
                    <button
                      type="button"
                      disabled={!canManage}
                      onClick={() => canManage && advance(item)}
                      className="mt-1"
                    >
                      <Badge variant={STATUS_VARIANT[item.status]}>
                        {t(`status${item.status}`)}
                      </Badge>
                    </button>
                  </div>
                ))}

                {/* Do column also surfaces in-progress generated drafts */}
                {phase === "DO"
                  ? drafts.map((d) => (
                      <Link
                        key={d.id}
                        href={`/generated/${d.id}`}
                        className="block rounded-md border border-dashed p-2 text-sm hover:border-primary"
                      >
                        {d.title}
                        <Badge variant="secondary" className="ms-2">
                          {d.status}
                        </Badge>
                      </Link>
                    ))
                  : null}

                {/* Check column surfaces reviews */}
                {phase === "CHECK"
                  ? reviews.map((r) => (
                      <Link
                        key={r.id}
                        href={`/reviews/${r.id}`}
                        className="block rounded-md border border-dashed p-2 text-sm hover:border-primary"
                      >
                        {r.packCode}
                        {r.score != null ? (
                          <Badge className="ms-2">{r.score}/100</Badge>
                        ) : (
                          <Badge variant="secondary" className="ms-2">
                            {r.status}
                          </Badge>
                        )}
                      </Link>
                    ))
                  : null}

                {/* Act column summarises improvement actions */}
                {phase === "ACT" && actionsCount > 0 ? (
                  <p className="text-xs text-muted-foreground">
                    {actionsCount} {t("actHint").toLowerCase()}
                  </p>
                ) : null}

                {items.length === 0 &&
                !(phase === "DO" && drafts.length) &&
                !(phase === "CHECK" && reviews.length) &&
                !(phase === "ACT" && actionsCount) ? (
                  <p className="text-xs italic text-muted-foreground">{t("noneYet")}</p>
                ) : null}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
