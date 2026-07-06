"use client";

import { useCallback, useEffect, useState } from "react";
import { useLocale, useTranslations } from "next-intl";

import { UploadDocumentDialog } from "@/components/documents/upload-document-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useRouter } from "@/i18n/navigation";

interface TemplateRow {
  id: string;
  name: string;
  formCode: string | null;
  language: string;
  version: number;
  createdAt: string;
  _count: { generatedDocuments: number };
}

export function TemplatesView({
  canManage,
  canGenerate,
  programs,
}: {
  canManage: boolean;
  canGenerate: boolean;
  programs: Array<{ id: string; code: string; name: string }>;
}) {
  const t = useTranslations("templates");
  const locale = useLocale();
  const router = useRouter();

  const [templates, setTemplates] = useState<TemplateRow[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [parsing, setParsing] = useState(false);

  const [templateId, setTemplateId] = useState("");
  const [programId, setProgramId] = useState(programs[0]?.id ?? "");
  const [title, setTitle] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const res = await fetch("/api/templates");
    if (res.ok) {
      const json = (await res.json()) as { templates: TemplateRow[] };
      setTemplates(json.templates);
      setLoaded(true);
      if (json.templates.length > 0) {
        setTemplateId((prev) => prev || json.templates[0].id);
      }
    }
  }, []);

  useEffect(() => {
    // setState happens only after the fetch resolves, not synchronously.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  // After a template upload, poll briefly while parsing completes.
  useEffect(() => {
    if (!parsing) return;
    const id = setInterval(async () => {
      void fetch("/api/jobs/run", { method: "POST" });
      const before = templates.length;
      await refresh();
      if (templates.length > before) setParsing(false);
    }, 3000);
    const stop = setTimeout(() => setParsing(false), 45000);
    return () => {
      clearInterval(id);
      clearTimeout(stop);
    };
  }, [parsing, refresh, templates.length]);

  async function createDocument(e: React.FormEvent) {
    e.preventDefault();
    setCreating(true);
    setError(null);
    const res = await fetch("/api/generated", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ templateId, programId, title, language: locale === "ar" ? "AR" : "EN" }),
    });
    if (!res.ok) {
      const json = (await res.json().catch(() => null)) as { error?: string } | null;
      setError(json?.error ?? "Failed");
      setCreating(false);
      return;
    }
    const { id } = (await res.json()) as { id: string };
    router.push(`/generated/${id}`);
  }

  return (
    <div className="space-y-6">
      {canManage ? (
        <UploadDocumentDialog
          defaultKind="TEMPLATE_SOURCE"
          onUploaded={() => {
            setParsing(true);
            void refresh();
          }}
        />
      ) : null}
      {parsing ? (
        <p className="text-sm text-muted-foreground">{t("parsing")}</p>
      ) : null}

      <Card>
        <CardContent className="p-0">
          {loaded && templates.length === 0 ? (
            <p className="p-6 text-sm text-muted-foreground">{t("empty")}</p>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="p-3 text-start font-medium">{t("name")}</th>
                  <th className="p-3 text-start font-medium">{t("formCode")}</th>
                  <th className="p-3 text-start font-medium">{t("version")}</th>
                  <th className="p-3 text-start font-medium">{t("generatedCount")}</th>
                </tr>
              </thead>
              <tbody>
                {templates.map((tpl) => (
                  <tr key={tpl.id} className="border-b last:border-0">
                    <td className="p-3 font-medium">{tpl.name}</td>
                    <td className="p-3">
                      {tpl.formCode ? <Badge variant="outline">{tpl.formCode}</Badge> : "—"}
                    </td>
                    <td className="p-3">v{tpl.version}</td>
                    <td className="p-3">{tpl._count.generatedDocuments}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>

      {canGenerate && templates.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t("newDocument")}</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={createDocument} className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="tpl">{t("selectTemplate")}</Label>
                <select
                  id="tpl"
                  value={templateId}
                  onChange={(e) => setTemplateId(e.target.value)}
                  className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
                >
                  {templates.map((tpl) => (
                    <option key={tpl.id} value={tpl.id}>
                      {tpl.formCode ? `${tpl.formCode} — ` : ""}
                      {tpl.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="prg">{t("selectProgram")}</Label>
                <select
                  id="prg"
                  value={programId}
                  onChange={(e) => setProgramId(e.target.value)}
                  className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
                >
                  {programs.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.code} — {p.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="space-y-2 sm:col-span-2">
                <Label htmlFor="doctitle">{t("documentTitle")}</Label>
                <Input
                  id="doctitle"
                  required
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="PHT-415 Course Specification 2026"
                />
              </div>
              {error ? (
                <p role="alert" className="text-sm text-destructive sm:col-span-2">
                  {error}
                </p>
              ) : null}
              <div className="sm:col-span-2">
                <Button type="submit" disabled={creating || !templateId || !programId}>
                  {creating ? t("creating") : t("create")}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
