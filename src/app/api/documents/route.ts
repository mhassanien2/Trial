import crypto from "node:crypto";

import { DocLanguage, DocumentKind, IngestStatus, PackOrigin } from "@prisma/client";
import { NextResponse } from "next/server";
import { z } from "zod";

import { logAudit } from "@/lib/audit";
import { prisma } from "@/lib/db";
import { enqueueJob, kickJobRunner } from "@/lib/jobs/queue";
import { ForbiddenError } from "@/lib/rbac";
import { getStorage } from "@/lib/storage";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";

const MAX_SIZE = 25 * 1024 * 1024; // 25 MB

const ALLOWED_MIMES = new Set([
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]);

// Kinds that go through the RAG ingestion pipeline.
const INGESTABLE = new Set<DocumentKind>([
  DocumentKind.STANDARDS_SOURCE,
  DocumentKind.REVIEW_SUBJECT,
  DocumentKind.EVIDENCE,
  DocumentKind.OTHER,
]);

const metaSchema = z.object({
  kind: z.nativeEnum(DocumentKind),
  title: z.string().min(1).max(300),
  language: z.nativeEnum(DocLanguage).default(DocLanguage.EN),
  programId: z.string().optional(),
  // For STANDARDS_SOURCE uploads: create a custom pack around the doc.
  packName: z.string().max(200).optional(),
  country: z.string().length(2).optional(),
});

export async function POST(req: Request) {
  try {
    const tenant = await requireTenantWith("documents.upload");

    const form = await req.formData();
    const file = form.get("file");
    if (!(file instanceof File)) {
      return NextResponse.json({ error: "Missing file" }, { status: 400 });
    }
    if (file.size === 0 || file.size > MAX_SIZE) {
      return NextResponse.json({ error: "File empty or exceeds 25MB" }, { status: 400 });
    }
    if (!ALLOWED_MIMES.has(file.type)) {
      return NextResponse.json(
        { error: `Unsupported file type: ${file.type}. Use PDF, DOCX or XLSX.` },
        { status: 400 }
      );
    }

    const meta = metaSchema.safeParse({
      kind: form.get("kind") ?? undefined,
      title: form.get("title") || file.name,
      language: form.get("language") ?? undefined,
      programId: form.get("programId") || undefined,
      packName: form.get("packName") || undefined,
      country: form.get("country") || undefined,
    });
    if (!meta.success) {
      return NextResponse.json(
        { error: "Invalid metadata", details: meta.error.flatten() },
        { status: 400 }
      );
    }

    // Program (if provided) must belong to the same institution.
    if (meta.data.programId) {
      const program = await prisma.program.findFirst({
        where: { id: meta.data.programId, institutionId: tenant.institutionId },
        select: { id: true },
      });
      if (!program) {
        return NextResponse.json({ error: "Unknown program" }, { status: 400 });
      }
    }

    const buffer = Buffer.from(await file.arrayBuffer());
    const sha256 = crypto.createHash("sha256").update(buffer).digest("hex");

    const docId = crypto.randomUUID();
    const safeName = file.name.replace(/[^\w.\-() ]+/g, "_").slice(0, 120);
    const storageKey = `${tenant.institutionId}/${docId}/${safeName}`;
    await getStorage().put(storageKey, buffer, file.type);

    const ingestable = INGESTABLE.has(meta.data.kind);
    const document = await prisma.document.create({
      data: {
        id: docId,
        institutionId: tenant.institutionId,
        programId: meta.data.programId ?? null,
        uploadedById: tenant.userId,
        kind: meta.data.kind,
        title: meta.data.title,
        language: meta.data.language,
        storageKey,
        mimeType: file.type,
        sizeBytes: file.size,
        sha256,
        ingestStatus: ingestable ? IngestStatus.PENDING : IngestStatus.NOT_APPLICABLE,
      },
    });

    // A standards upload becomes a selectable custom pack.
    let packId: string | null = null;
    if (meta.data.kind === DocumentKind.STANDARDS_SOURCE) {
      const pack = await prisma.standardsPack.create({
        data: {
          institutionId: tenant.institutionId,
          origin: PackOrigin.CUSTOM,
          country: meta.data.country ?? "XX",
          code: `CUSTOM-${docId.slice(0, 8).toUpperCase()}`,
          nameEn: meta.data.packName ?? meta.data.title,
          version: "1.0",
          sourceDocumentId: document.id,
        },
      });
      packId = pack.id;
    }

    if (ingestable) {
      await enqueueJob("ingest_document", { documentId: document.id }, tenant.institutionId);
      kickJobRunner();
    }

    await logAudit({
      institutionId: tenant.institutionId,
      userId: tenant.userId,
      action: "document.upload",
      entityType: "Document",
      entityId: document.id,
      metadata: { kind: meta.data.kind, sha256, sizeBytes: file.size, packId },
    });

    return NextResponse.json(
      { id: document.id, ingestStatus: document.ingestStatus, packId },
      { status: 201 }
    );
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[documents:POST]", err);
    return NextResponse.json({ error: "Upload failed" }, { status: 500 });
  }
}

export async function GET() {
  try {
    const tenant = await requireTenantWith("documents.view");
    const documents = await prisma.document.findMany({
      where: { institutionId: tenant.institutionId },
      orderBy: { createdAt: "desc" },
      take: 100,
      select: {
        id: true,
        title: true,
        kind: true,
        language: true,
        mimeType: true,
        sizeBytes: true,
        ingestStatus: true,
        ingestError: true,
        pageCount: true,
        createdAt: true,
        _count: { select: { chunks: true } },
      },
    });
    return NextResponse.json({ documents });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[documents:GET]", err);
    return NextResponse.json({ error: "Failed to list documents" }, { status: 500 });
  }
}
