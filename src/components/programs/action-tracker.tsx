"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const STATUS_CYCLE: Record<string, string> = {
  OPEN: "IN_PROGRESS",
  IN_PROGRESS: "COMPLETED",
  COMPLETED: "OPEN",
  CANCELLED: "OPEN",
};

const STATUS_VARIANT: Record<string, "default" | "secondary" | "outline" | "destructive"> = {
  COMPLETED: "default",
  IN_PROGRESS: "secondary",
  OPEN: "outline",
  CANCELLED: "destructive",
};

interface Action {
  id: string;
  title: string;
  status: string;
  ownerLabel: string | null;
  dueDate: string | null;
  fromFinding: boolean;
}

export function ActionTracker({
  programId,
  canManage,
  members,
  actions,
}: {
  programId: string;
  canManage: boolean;
  members: Array<{ id: string; label: string }>;
  actions: Action[];
}) {
  const t = useTranslations("actions");
  const router = useRouter();
  const [adding, setAdding] = useState(false);
  const [title, setTitle] = useState("");
  const [ownerId, setOwnerId] = useState("");
  const [dueDate, setDueDate] = useState("");
  const [pending, setPending] = useState(false);

  async function create(e: React.FormEvent) {
    e.preventDefault();
    setPending(true);
    await fetch(`/api/programs/${programId}/actions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title,
        ownerId: ownerId || undefined,
        dueDate: dueDate ? new Date(dueDate).toISOString() : undefined,
      }),
    });
    setTitle("");
    setOwnerId("");
    setDueDate("");
    setAdding(false);
    setPending(false);
    router.refresh();
  }

  async function cycle(action: Action) {
    await fetch(`/api/programs/${programId}/actions`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actionId: action.id, status: STATUS_CYCLE[action.status] }),
    });
    router.refresh();
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-lg">{t("title")}</CardTitle>
        {canManage ? (
          <Button size="sm" onClick={() => setAdding(!adding)}>
            {t("add")}
          </Button>
        ) : null}
      </CardHeader>
      <CardContent className="space-y-4">
        {adding ? (
          <form onSubmit={create} className="grid gap-3 rounded-md border p-3 sm:grid-cols-3">
            <div className="space-y-1 sm:col-span-3">
              <Label htmlFor="atitle">{t("actionTitle")}</Label>
              <Input
                id="atitle"
                required
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="owner">{t("owner")}</Label>
              <select
                id="owner"
                value={ownerId}
                onChange={(e) => setOwnerId(e.target.value)}
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
              >
                <option value="">{t("unassigned")}</option>
                {members.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <Label htmlFor="due">{t("dueDate")}</Label>
              <Input
                id="due"
                type="date"
                value={dueDate}
                onChange={(e) => setDueDate(e.target.value)}
              />
            </div>
            <div className="flex items-end">
              <Button type="submit" disabled={pending}>
                {pending ? t("creating") : t("create")}
              </Button>
            </div>
          </form>
        ) : null}

        {actions.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("empty")}</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="p-2 text-start font-medium">{t("actionTitle")}</th>
                  <th className="p-2 text-start font-medium">{t("owner")}</th>
                  <th className="p-2 text-start font-medium">{t("dueDate")}</th>
                  <th className="p-2 text-start font-medium">{t("status")}</th>
                </tr>
              </thead>
              <tbody>
                {actions.map((a) => (
                  <tr key={a.id} className="border-b last:border-0">
                    <td className="p-2">
                      {a.title}
                      {a.fromFinding ? (
                        <Badge variant="outline" className="ms-2">
                          {t("fromFinding")}
                        </Badge>
                      ) : null}
                    </td>
                    <td className="p-2 text-muted-foreground">
                      {a.ownerLabel ?? t("unassigned")}
                    </td>
                    <td className="p-2 text-muted-foreground">
                      {a.dueDate ? new Date(a.dueDate).toLocaleDateString() : "—"}
                    </td>
                    <td className="p-2">
                      <button
                        type="button"
                        disabled={!canManage}
                        onClick={() => canManage && cycle(a)}
                      >
                        <Badge variant={STATUS_VARIANT[a.status]}>{t(a.status)}</Badge>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
