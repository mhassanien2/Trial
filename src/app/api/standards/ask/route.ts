import crypto from "node:crypto";

import { NextResponse } from "next/server";
import { z } from "zod";

import { AI_MODEL, getAnthropic, isAiEnabled } from "@/lib/ai/client";
import {
  PROMPT_VERSION,
  buildSystemPrompt,
  buildUserPrompt,
  type QaContextChunk,
} from "@/lib/ai/prompts/standards-qa";
import { prisma } from "@/lib/db";
import { hybridSearch } from "@/lib/rag/retrieval";
import { LIMITS, rateLimit, tooManyRequests } from "@/lib/rate-limit";
import { ForbiddenError } from "@/lib/rbac";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";
export const maxDuration = 60;

const bodySchema = z.object({
  question: z.string().min(3).max(2000),
  documentId: z.string().optional(),
  packId: z.string().optional(),
  locale: z.enum(["en", "ar"]).default("en"),
});

/**
 * Cited standards Q&A. Non-negotiables enforced here:
 * - NEVER answers without retrieved context (low confidence ⇒ says so).
 * - Every AI call is logged to AiInteraction (who/when/model/prompt
 *   version/input hash/output) for accreditation traceability.
 */
export async function POST(req: Request) {
  try {
    const tenant = await requireTenantWith("standards.view");

    const rl = rateLimit(tenant.userId, "ai", LIMITS.ai);
    if (!rl.ok) return tooManyRequests(rl);

    const body = bodySchema.safeParse(await req.json());
    if (!body.success) {
      return NextResponse.json(
        { error: "Invalid request", details: body.error.flatten() },
        { status: 400 }
      );
    }
    const { question, documentId, packId, locale } = body.data;

    // Scope: explicit document, a custom pack's source document, or all
    // of the institution's ingested documents.
    let documentIds: string[] | undefined;
    if (documentId) {
      documentIds = [documentId];
    } else if (packId) {
      const pack = await prisma.standardsPack.findFirst({
        where: {
          id: packId,
          OR: [{ institutionId: tenant.institutionId }, { institutionId: null }],
        },
        select: { sourceDocumentId: true },
      });
      if (!pack) {
        return NextResponse.json({ error: "Unknown pack" }, { status: 404 });
      }
      if (pack.sourceDocumentId) documentIds = [pack.sourceDocumentId];
      // Official packs have no source document (placeholder structure);
      // fall through to all institution documents.
    }

    const { chunks, confident } = await hybridSearch({
      institutionId: tenant.institutionId,
      query: question,
      documentIds,
      limit: 8,
    });

    const sources = chunks.map((c, i) => ({
      index: i + 1,
      chunkId: c.chunkId,
      documentId: c.documentId,
      documentTitle: c.documentTitle,
      page: c.page,
      headingPath: c.headingPath,
      excerpt: c.content.slice(0, 400),
    }));

    // Rule: never answer standards questions without retrieved context.
    if (!confident || chunks.length === 0) {
      return NextResponse.json({
        answer: null,
        lowConfidence: true,
        aiEnabled: isAiEnabled(),
        sources,
      });
    }

    if (!isAiEnabled()) {
      // Retrieval-only mode: return cited passages, no synthesis.
      await logInteraction(tenant, question, chunks.length, null, "retrieval-only");
      return NextResponse.json({
        answer: null,
        aiDisabled: true,
        lowConfidence: false,
        sources,
      });
    }

    const contextChunks: QaContextChunk[] = chunks.map((c, i) => ({
      index: i + 1,
      documentTitle: c.documentTitle,
      page: c.page,
      headingPath: c.headingPath,
      content: c.content,
    }));

    const system = buildSystemPrompt(locale);
    const user = buildUserPrompt(question, contextChunks);
    const started = Date.now();

    const response = await getAnthropic().messages.create({
      model: AI_MODEL,
      max_tokens: 1500,
      system,
      messages: [{ role: "user", content: user }],
    });

    const answer = response.content
      .filter((b) => b.type === "text")
      .map((b) => b.text)
      .join("\n");

    await prisma.aiInteraction.create({
      data: {
        institutionId: tenant.institutionId,
        userId: tenant.userId,
        feature: "qa_chat",
        model: AI_MODEL,
        promptVersion: PROMPT_VERSION,
        inputHash: crypto
          .createHash("sha256")
          .update(system + "\n" + user)
          .digest("hex"),
        outputText: answer,
        citations: sources,
        tokensIn: response.usage.input_tokens,
        tokensOut: response.usage.output_tokens,
        latencyMs: Date.now() - started,
      },
    });

    return NextResponse.json({ answer, lowConfidence: false, sources });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[standards:ask]", err);
    return NextResponse.json({ error: "Question failed" }, { status: 500 });
  }
}

async function logInteraction(
  tenant: { institutionId: string; userId: string },
  question: string,
  chunkCount: number,
  output: string | null,
  model: string
) {
  await prisma.aiInteraction.create({
    data: {
      institutionId: tenant.institutionId,
      userId: tenant.userId,
      feature: "qa_chat",
      model,
      promptVersion: PROMPT_VERSION,
      inputHash: crypto.createHash("sha256").update(question).digest("hex"),
      outputText: output,
      citations: { chunkCount },
    },
  });
}
