import { IngestStatus } from "@prisma/client";
import { z } from "zod";

import { prisma } from "@/lib/db";
import { getStorage } from "@/lib/storage";
import { parseDocxTemplate } from "@/lib/templates/parser";

const payloadSchema = z.object({ documentId: z.string() });

const DOCX_MIME =
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document";

/** Parses an uploaded DOCX template into a Template row (schemaJson). */
export async function parseTemplateHandler(rawPayload: unknown): Promise<void> {
  const { documentId } = payloadSchema.parse(rawPayload);
  const doc = await prisma.document.findUniqueOrThrow({ where: { id: documentId } });

  await prisma.document.update({
    where: { id: doc.id },
    data: { ingestStatus: IngestStatus.PROCESSING, ingestError: null },
  });

  try {
    if (doc.mimeType !== DOCX_MIME) {
      throw new Error("Templates must be DOCX files");
    }
    const buffer = await getStorage().get(doc.storageKey);
    const schema = await parseDocxTemplate(buffer, {
      title: doc.title,
      language: doc.language === "AR" ? "AR" : "EN",
    });

    const meta = (doc.metadata ?? {}) as { formCode?: string };
    await prisma.template.upsert({
      where: { sourceDocumentId: doc.id },
      update: { schemaJson: schema, name: doc.title, version: { increment: 1 } },
      create: {
        institutionId: doc.institutionId,
        sourceDocumentId: doc.id,
        name: doc.title,
        formCode: meta.formCode ?? null,
        language: doc.language,
        schemaJson: schema,
      },
    });

    await prisma.document.update({
      where: { id: doc.id },
      data: { ingestStatus: IngestStatus.READY },
    });
  } catch (err) {
    await prisma.document.update({
      where: { id: doc.id },
      data: {
        ingestStatus: IngestStatus.FAILED,
        ingestError: (err instanceof Error ? err.message : String(err)).slice(0, 2000),
      },
    });
    throw err;
  }
}
