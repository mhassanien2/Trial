"use client";

import { useState } from "react";
import { Link2, X } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { useLocale, useTranslations } from "next-intl";

import { UploadDocumentDialog } from "@/components/documents/upload-document-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface CriterionCoverage {
  id: string;
  code: string;
  title: string;
  links: Array<{ id: string; documentId: string; documentTitle: string }>;
}
interface StandardCoverage {
  code: string;
  title: string;
  criteria: CriterionCoverage[];
}
interface Coverage {
  packName: string;
  standards: StandardCoverage[];
}

// Colour intensity by evidence count (0 → muted, 3+ → strong primary).
function cellClass(count: number): string {
  if (count === 0) return "bg-muted text-muted-foreground";
  if (count === 1) return "bg-primary/20";
  if (count === 2) return "bg-primary/45 text-primary-foreground";
  return "bg-primary/80 text-primary-foreground";
}

export function EvidenceHeatMap({
  programId,
  canManage,
  packs,
  selectedPackId,
  evidenceDocs,
  coverage,
}: {
  programId: string;
  canManage: boolean;
  packs: Array<{ id: string; label: string }>;
  selectedPackId: string | null;
  evidenceDocs: Array<{ id: string; title: string }>;
  coverage: Coverage | null;
}) {
  const t = useTranslations("evidence");
  const locale = useLocale();
  const router = useRouter();
  const searchParams = useSearchParams();

  const [active, setActive] = useState<string | null>(null);
  const [docId, setDocId] = useState(evidenceDocs[0]?.id ?? "");

  function switchPack(packId: string) {
    const p = new URLSearchParams(searchParams.toString());
    p.set("pack", packId);
    router.push(`/${locale}/programs/${programId}?${p.toString()}`);
  }

  async function link(criterionId: string) {
    if (!docId) return;
    await fetch(`/api/programs/${programId}/evidence`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ documentId: docId, criterionId }),
    });
    setActive(null);
    router.refresh();
  }

  async function unlink(linkId: string) {
    await fetch(`/api/programs/${programId}/evidence`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ linkId }),
    });
    router.refresh();
  }

  const allCriteria = coverage?.standards.flatMap((s) => s.criteria) ?? [];
  const covered = allCriteria.filter((c) => c.links.length > 0).length;
  const total = allCriteria.length;

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <CardTitle className="text-lg">{t("title")}</CardTitle>
            <p className="text-sm text-muted-foreground">{t("subtitle")}</p>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={selectedPackId ?? ""}
              onChange={(e) => switchPack(e.target.value)}
              className="h-9 rounded-md border border-input bg-transparent px-3 text-sm shadow-sm"
            >
              {packs.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
            {canManage ? (
              <UploadDocumentDialog
                defaultKind="EVIDENCE"
                compact
                onUploaded={() => router.refresh()}
              />
            ) : null}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {total > 0 ? (
          <div className="flex items-center gap-4 text-sm">
            <span>
              <span className="font-semibold text-primary">{covered}</span>
              <span className="text-muted-foreground"> / {total} {t("covered")}</span>
            </span>
            <span className="text-muted-foreground">
              {total - covered} {t("gaps")}
            </span>
            {/* legend */}
            <span className="ms-auto flex items-center gap-1 text-xs text-muted-foreground">
              {t("legend")}:
              {[0, 1, 2, 3].map((n) => (
                <span key={n} className={`inline-block h-4 w-6 rounded ${cellClass(n)}`} />
              ))}
            </span>
          </div>
        ) : null}

        {!coverage ? (
          <p className="text-sm text-muted-foreground">—</p>
        ) : (
          <div className="space-y-4">
            {coverage.standards.map((s) => (
              <div key={s.code}>
                <p className="mb-2 text-sm font-semibold">
                  <span className="me-2 text-muted-foreground">{s.code}</span>
                  {s.title}
                </p>
                <div className="flex flex-wrap gap-2">
                  {s.criteria.map((c) => {
                    const count = c.links.length;
                    return (
                      <div key={c.id} className="relative">
                        <button
                          type="button"
                          onClick={() =>
                            canManage ? setActive(active === c.id ? null : c.id) : undefined
                          }
                          title={`${c.code}${c.title ? ` — ${c.title}` : ""} · ${
                            count > 0 ? t("linkedCount", { count }) : t("none")
                          }`}
                          className={`flex h-14 w-20 flex-col items-center justify-center rounded-md border text-xs transition ${cellClass(
                            count
                          )} ${canManage ? "cursor-pointer hover:ring-2 hover:ring-ring" : ""}`}
                        >
                          <span className="font-medium">{c.code}</span>
                          <span>{count > 0 ? `● ${count}` : "○"}</span>
                        </button>

                        {active === c.id ? (
                          <div className="absolute z-10 mt-1 w-64 rounded-md border bg-popover p-3 shadow-md">
                            <p className="mb-2 text-xs font-semibold">{t("linkTo")}</p>
                            {c.links.length > 0 ? (
                              <ul className="mb-2 space-y-1">
                                {c.links.map((l) => (
                                  <li
                                    key={l.id}
                                    className="flex items-center justify-between gap-1 text-xs"
                                  >
                                    <span className="truncate">{l.documentTitle}</span>
                                    <button
                                      type="button"
                                      onClick={() => unlink(l.id)}
                                      aria-label={t("remove")}
                                    >
                                      <X className="h-3 w-3" />
                                    </button>
                                  </li>
                                ))}
                              </ul>
                            ) : null}
                            {evidenceDocs.length === 0 ? (
                              <p className="text-xs text-muted-foreground">
                                {t("noEvidenceDocs")}
                              </p>
                            ) : (
                              <div className="flex gap-1">
                                <select
                                  value={docId}
                                  onChange={(e) => setDocId(e.target.value)}
                                  className="h-8 flex-1 rounded-md border border-input bg-transparent px-2 text-xs"
                                >
                                  {evidenceDocs.map((d) => (
                                    <option key={d.id} value={d.id}>
                                      {d.title}
                                    </option>
                                  ))}
                                </select>
                                <Button size="sm" className="h-8" onClick={() => link(c.id)}>
                                  <Link2 className="h-3 w-3" />
                                </Button>
                              </div>
                            )}
                          </div>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
