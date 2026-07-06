import { notFound } from "next/navigation";
import { getTranslations } from "next-intl/server";

import { PdcaBoard } from "@/components/programs/pdca-board";
import { ActionTracker } from "@/components/programs/action-tracker";
import { EvidenceHeatMap } from "@/components/programs/evidence-heatmap";
import { ReadinessTrend } from "@/components/programs/readiness-trend";
import { MockPanel } from "@/components/programs/mock-panel";
import { CrossMapper } from "@/components/programs/cross-mapper";
import { Badge } from "@/components/ui/badge";
import { prisma } from "@/lib/db";
import { can } from "@/lib/rbac";
import { isPlaceholder } from "@/lib/standards/schema";
import { requireTenant } from "@/lib/tenant";

export const dynamic = "force-dynamic";

export default async function ProgramWorkspacePage({
  params,
  searchParams,
}: {
  params: Promise<{ locale: string; id: string }>;
  searchParams: Promise<{ pack?: string }>;
}) {
  const { locale, id } = await params;
  const { pack: packParam } = await searchParams;
  const tenant = await requireTenant(locale);
  const t = await getTranslations();
  const ar = locale === "ar";

  const program = await prisma.program.findFirst({
    where: { id, institutionId: tenant.institutionId },
    select: { id: true, code: true, nameEn: true, nameAr: true, degreeLevel: true },
  });
  if (!program) notFound();

  const [planItems, drafts, reviews, actions, members, latest, snapshots, ssrDocs, latestPanel] =
    await Promise.all([
    prisma.planItem.findMany({
      where: { programId: id },
      orderBy: { createdAt: "asc" },
    }),
    prisma.generatedDocument.findMany({
      where: { programId: id, status: { in: ["DRAFT", "IN_REVIEW"] } },
      select: { id: true, title: true, status: true },
      orderBy: { updatedAt: "desc" },
    }),
    prisma.review.findMany({
      where: { programId: id },
      orderBy: { createdAt: "desc" },
      select: {
        id: true,
        status: true,
        readinessScore: true,
        pack: { select: { code: true } },
      },
    }),
    prisma.improvementAction.findMany({
      where: { programId: id },
      orderBy: { createdAt: "desc" },
      include: { owner: { select: { name: true, email: true } } },
    }),
    prisma.user.findMany({
      where: { institutionId: tenant.institutionId, isActive: true },
      select: { id: true, name: true, email: true },
      orderBy: { name: "asc" },
    }),
    prisma.readinessSnapshot.findFirst({
      where: { programId: id },
      orderBy: { createdAt: "desc" },
    }),
    prisma.readinessSnapshot.findMany({
      where: { programId: id },
      orderBy: { createdAt: "asc" },
      select: { score: true, createdAt: true },
    }),
    prisma.document.findMany({
      where: {
        institutionId: tenant.institutionId,
        ingestStatus: "READY",
        kind: { in: ["REVIEW_SUBJECT", "OTHER"] },
      },
      select: { id: true, title: true },
      orderBy: { createdAt: "desc" },
    }),
    prisma.mockPanelRun.findFirst({
      where: { programId: id },
      orderBy: { createdAt: "desc" },
      select: { id: true },
    }),
  ]);

  // Standards packs + evidence coverage for the heat map.
  const packs = await prisma.standardsPack.findMany({
    where: {
      isActive: true,
      OR: [{ institutionId: null }, { institutionId: tenant.institutionId }],
    },
    select: { id: true, nameEn: true, nameAr: true, code: true },
    orderBy: [{ origin: "asc" }, { code: "asc" }],
  });
  const selectedPackId = packParam ?? packs[0]?.id ?? null;

  const evidenceDocs = await prisma.document.findMany({
    where: { institutionId: tenant.institutionId, kind: "EVIDENCE" },
    select: { id: true, title: true },
    orderBy: { createdAt: "desc" },
  });

  let coverage: {
    packName: string;
    standards: Array<{
      code: string;
      title: string;
      criteria: Array<{
        id: string;
        code: string;
        title: string;
        links: Array<{ id: string; documentId: string; documentTitle: string }>;
      }>;
    }>;
  } | null = null;

  if (selectedPackId) {
    const pack = await prisma.standardsPack.findFirst({
      where: {
        id: selectedPackId,
        OR: [{ institutionId: null }, { institutionId: tenant.institutionId }],
      },
      include: {
        standards: {
          orderBy: { sortOrder: "asc" },
          include: { criteria: { orderBy: { sortOrder: "asc" } } },
        },
      },
    });
    if (pack) {
      const links = await prisma.evidenceLink.findMany({
        where: { programId: id, criterion: { standard: { packId: pack.id } } },
        select: {
          id: true,
          criterionId: true,
          documentId: true,
          document: { select: { title: true } },
        },
      });
      const byCriterion = new Map<string, typeof links>();
      for (const l of links) {
        const arr = byCriterion.get(l.criterionId) ?? [];
        arr.push(l);
        byCriterion.set(l.criterionId, arr);
      }
      coverage = {
        packName: ar && pack.nameAr ? pack.nameAr : pack.nameEn,
        standards: pack.standards.map((s) => ({
          code: s.code,
          title: isPlaceholder(s.titleEn) ? s.code : s.titleEn,
          criteria: s.criteria.map((c) => ({
            id: c.id,
            code: c.code,
            title: isPlaceholder(c.titleEn) ? "" : c.titleEn,
            links: (byCriterion.get(c.id) ?? []).map((l) => ({
              id: l.id,
              documentId: l.documentId,
              documentTitle: l.document.title,
            })),
          })),
        })),
      };
    }
  }

  const canManageActions = can(tenant.role, "actions.manage");
  const canManageEvidence = can(tenant.role, "evidence.manage");

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight">
              {ar ? program.nameAr : program.nameEn}
            </h1>
            <Badge variant="outline">{program.code}</Badge>
          </div>
          <p className="text-sm text-muted-foreground">{t("pdca.workspace")}</p>
        </div>
        {latest ? (
          <div className="text-end">
            <div className="text-3xl font-bold">{latest.score}/100</div>
            <div className="text-xs text-muted-foreground">{t("programs.readiness")}</div>
          </div>
        ) : null}
      </div>

      <ReadinessTrend
        points={snapshots.map((s) => ({
          score: s.score,
          createdAt: s.createdAt.toISOString(),
        }))}
      />

      <PdcaBoard
        programId={id}
        canManage={canManageActions}
        planItems={planItems.map((p) => ({
          id: p.id,
          phase: p.phase,
          title: p.title,
          status: p.status,
        }))}
        drafts={drafts}
        reviews={reviews.map((r) => ({
          id: r.id,
          status: r.status,
          score: r.readinessScore,
          packCode: r.pack.code,
        }))}
        actionsCount={actions.length}
      />

      <ActionTracker
        programId={id}
        canManage={canManageActions}
        members={members.map((m) => ({ id: m.id, label: m.name ?? m.email }))}
        actions={actions.map((a) => ({
          id: a.id,
          title: a.title,
          status: a.status,
          ownerLabel: a.owner?.name ?? a.owner?.email ?? null,
          dueDate: a.dueDate?.toISOString() ?? null,
          fromFinding: Boolean(a.findingId),
        }))}
      />

      <EvidenceHeatMap
        programId={id}
        canManage={canManageEvidence}
        packs={packs.map((p) => ({
          id: p.id,
          label: `${p.code} — ${ar && p.nameAr ? p.nameAr : p.nameEn}`,
        }))}
        selectedPackId={selectedPackId}
        evidenceDocs={evidenceDocs}
        coverage={coverage}
      />

      <MockPanel
        programId={id}
        canRun={can(tenant.role, "reviews.run")}
        ssrDocs={ssrDocs}
        latestRunId={latestPanel?.id ?? null}
      />

      <CrossMapper
        programId={id}
        packs={packs.map((p) => ({
          id: p.id,
          label: `${p.code} — ${ar && p.nameAr ? p.nameAr : p.nameEn}`,
        }))}
      />
    </div>
  );
}
