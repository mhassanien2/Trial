import crypto from "node:crypto";

import { NextResponse } from "next/server";
import { z } from "zod";

import { AI_MODEL, getAnthropic, isAiEnabled } from "@/lib/ai/client";
import {
  NOT_ENOUGH_INFO,
  PROMPT_VERSION,
  buildSystemPrompt,
  buildUserPrompt,
} from "@/lib/ai/prompts/template-fill";
import { prisma } from "@/lib/db";
import { hybridSearch } from "@/lib/rag/retrieval";
import { ForbiddenError } from "@/lib/rbac";
import { collectFields, templateSchemaSchema } from "@/lib/templates/schema";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";
export const maxDuration = 60;

const bodySchema = z.object({
  fieldId: z.string(),
  locale: z.enum(["en", "ar"]).default("en"),
});

/**
 * AI-assisted filling for one template field. Uses RAG context from the
 * institution's ingested documents + program facts. Returns
 * { suggestion: null, notEnoughInfo: true } when the model cannot draft
 * from context — the wizard then leaves the field for manual input
 * (and exports fall back to [REQUIRES INPUT]).
 */
export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const tenant = await requireTenantWith("generator.use");
    const body = bodySchema.safeParse(await req.json());
    if (!body.success) {
      return NextResponse.json({ error: "Invalid request" }, { status: 400 });
    }

    if (!isAiEnabled()) {
      return NextResponse.json({ suggestion: null, aiDisabled: true });
    }

    const doc = await prisma.generatedDocument.findFirst({
      where: { id, institutionId: tenant.institutionId },
      include: { template: true, program: true },
    });
    if (!doc) return NextResponse.json({ error: "Not found" }, { status: 404 });

    const schema = templateSchemaSchema.parse(doc.template.schemaJson);
    const fieldInfo = collectFields(schema).find((f) => f.field.id === body.data.fieldId);
    if (!fieldInfo) {
      return NextResponse.json({ error: "Unknown field" }, { status: 400 });
    }

    const { chunks } = await hybridSearch({
      institutionId: tenant.institutionId,
      query: `${fieldInfo.sectionHeading} ${fieldInfo.field.label} ${doc.program.nameEn}`,
      limit: 6,
    });

    const system = buildSystemPrompt(body.data.locale);
    const user = buildUserPrompt({
      sectionHeading: fieldInfo.sectionHeading,
      fieldLabel: fieldInfo.field.label,
      documentTitle: doc.template.name,
      program: {
        name: doc.program.nameEn,
        code: doc.program.code,
        degreeLevel: doc.program.degreeLevel,
        department: doc.program.department,
      },
      chunks: chunks.map((c, i) => ({
        index: i + 1,
        documentTitle: c.documentTitle,
        page: c.page,
        content: c.content,
      })),
    });

    const started = Date.now();
    const response = await getAnthropic().messages.create({
      model: AI_MODEL,
      max_tokens: 800,
      system,
      messages: [{ role: "user", content: user }],
    });
    const text = response.content
      .filter((b) => b.type === "text")
      .map((b) => b.text)
      .join("\n")
      .trim();

    const notEnough = text === NOT_ENOUGH_INFO || text.length === 0;

    await prisma.aiInteraction.create({
      data: {
        institutionId: tenant.institutionId,
        userId: tenant.userId,
        feature: "generator",
        model: AI_MODEL,
        promptVersion: PROMPT_VERSION,
        inputHash: crypto.createHash("sha256").update(system + "\n" + user).digest("hex"),
        outputText: text,
        citations: chunks.map((c, i) => ({
          index: i + 1,
          documentId: c.documentId,
          documentTitle: c.documentTitle,
          page: c.page,
        })),
        tokensIn: response.usage.input_tokens,
        tokensOut: response.usage.output_tokens,
        latencyMs: Date.now() - started,
      },
    });

    return NextResponse.json({
      suggestion: notEnough ? null : text,
      notEnoughInfo: notEnough,
      sources: chunks.map((c, i) => ({
        index: i + 1,
        documentTitle: c.documentTitle,
        page: c.page,
      })),
    });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[generated:suggest]", err);
    return NextResponse.json({ error: "Suggestion failed" }, { status: 500 });
  }
}
