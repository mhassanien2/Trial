import crypto from "node:crypto";

import { IngestStatus, Prisma } from "@prisma/client";
import { z } from "zod";

import { embedTexts } from "@/lib/ai/embeddings";
import { prisma } from "@/lib/db";
import { chunkPages } from "@/lib/ingest/chunk";
import { extractText } from "@/lib/ingest/extract";
import { getStorage } from "@/lib/storage";

const payloadSchema = z.object({ documentId: z.string() });

const EMBED_BATCH = 32;

/**
 * Ingestion pipeline: storage → text extraction → heading-aware chunking
 * → embeddings → pgvector rows. Marks the Document READY/FAILED.
 */
export async function ingestDocumentHandler(rawPayload: unknown): Promise<void> {
  const { documentId } = payloadSchema.parse(rawPayload);

  const doc = await prisma.document.findUniqueOrThrow({
    where: { id: documentId },
  });

  await prisma.document.update({
    where: { id: doc.id },
    data: { ingestStatus: IngestStatus.PROCESSING, ingestError: null },
  });

  try {
    const buffer = await getStorage().get(doc.storageKey);
    const { pages, pageCount } = await extractText(buffer, doc.mimeType);
    const chunks = chunkPages(pages);
    if (chunks.length === 0) {
      throw new Error("No extractable text found in document");
    }

    // Re-ingest safety: drop existing chunks for this document.
    await prisma.documentChunk.deleteMany({ where: { documentId: doc.id } });

    for (let i = 0; i < chunks.length; i += EMBED_BATCH) {
      const batch = chunks.slice(i, i + EMBED_BATCH);
      const embeddings = await embedTexts(batch.map((c) => c.content));

      // Raw insert: Prisma cannot bind Unsupported("vector") columns.
      for (let j = 0; j < batch.length; j++) {
        const c = batch[j];
        const vector = `[${embeddings[j].join(",")}]`;
        await prisma.$executeRaw`
          INSERT INTO "DocumentChunk"
            (id, "documentId", "chunkIndex", content, embedding, page,
             "headingPath", "criterionCode", "tokenCount", metadata)
          VALUES
            (${crypto.randomUUID()}, ${doc.id}, ${i + j}, ${c.content},
             ${vector}::vector, ${c.page}, ${c.headingPath},
             ${c.criterionCode}, ${Math.ceil(c.content.length / 4)},
             ${Prisma.JsonNull})
        `;
      }
    }

    await prisma.document.update({
      where: { id: doc.id },
      data: { ingestStatus: IngestStatus.READY, pageCount },
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
