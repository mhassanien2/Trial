"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { UploadDocumentDialog } from "./upload-document-dialog";

export interface DocumentRow {
  id: string;
  title: string;
  kind: string;
  language: string;
  mimeType: string;
  sizeBytes: number;
  ingestStatus: string;
  ingestError: string | null;
  pageCount: number | null;
  createdAt: string;
  _count: { chunks: number };
}

const STATUS_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  READY: "default",
  PROCESSING: "secondary",
  PENDING: "secondary",
  FAILED: "destructive",
  NOT_APPLICABLE: "outline",
};

export function DocumentsView({ canUpload }: { canUpload: boolean }) {
  const t = useTranslations("documents");
  const [docs, setDocs] = useState<DocumentRow[]>([]);
  const [loaded, setLoaded] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(async () => {
    const res = await fetch("/api/documents");
    if (res.ok) {
      const json = (await res.json()) as { documents: DocumentRow[] };
      setDocs(json.documents);
      setLoaded(true);
      // Keep polling while anything is still ingesting; also nudge the
      // job runner so queued work makes progress in dev.
      const busy = json.documents.some((d) =>
        ["PENDING", "PROCESSING"].includes(d.ingestStatus)
      );
      if (timer.current) clearTimeout(timer.current);
      if (busy) {
        void fetch("/api/jobs/run", { method: "POST" });
        timer.current = setTimeout(refresh, 3500);
      }
    }
  }, []);

  useEffect(() => {
    void refresh();
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [refresh]);

  return (
    <div className="space-y-4">
      {canUpload ? <UploadDocumentDialog onUploaded={refresh} /> : null}

      <Card>
        <CardContent className="p-0">
          {loaded && docs.length === 0 ? (
            <p className="p-6 text-sm text-muted-foreground">{t("empty")}</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-start text-muted-foreground">
                    <th className="p-3 text-start font-medium">{t("name")}</th>
                    <th className="p-3 text-start font-medium">{t("kind")}</th>
                    <th className="p-3 text-start font-medium">{t("status")}</th>
                    <th className="p-3 text-start font-medium">{t("pages")}</th>
                    <th className="p-3 text-start font-medium">{t("chunks")}</th>
                    <th className="p-3 text-start font-medium">{t("uploaded")}</th>
                  </tr>
                </thead>
                <tbody>
                  {docs.map((d) => (
                    <tr key={d.id} className="border-b last:border-0">
                      <td className="max-w-[280px] truncate p-3 font-medium">
                        {d.title}
                        {d.ingestError ? (
                          <p className="truncate text-xs text-destructive">
                            {d.ingestError}
                          </p>
                        ) : null}
                      </td>
                      <td className="p-3">{t(`kinds.${d.kind}`)}</td>
                      <td className="p-3">
                        <Badge variant={STATUS_VARIANT[d.ingestStatus] ?? "outline"}>
                          {t(`statuses.${d.ingestStatus}`)}
                        </Badge>
                      </td>
                      <td className="p-3">{d.pageCount ?? "—"}</td>
                      <td className="p-3">{d._count.chunks || "—"}</td>
                      <td className="p-3 text-muted-foreground">
                        {new Date(d.createdAt).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
