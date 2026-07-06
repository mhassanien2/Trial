import crypto from "node:crypto";

import { Prisma, ReviewStatus } from "@prisma/client";

import { AI_MODEL, getAnthropic, isAiEnabled } from "@/lib/ai/client";
import {
  PERSONAS,
  PROMPT_VERSION,
  buildSystemPrompt,
  buildUserPrompt,
  type PanelContextChunk,
  type PersonaKey,
} from "@/lib/ai/prompts/mock-panel";
import { prisma } from "@/lib/db";
import { hybridSearch } from "@/lib/rag/retrieval";

export interface PersonaResult {
  key: PersonaKey;
  nameEn: string;
  nameAr: string;
  critique: string;
  strengths: string[];
  concerns: string[];
  questions: Array<{ question: string; suggestedAnswer: string }>;
  aiDisabled?: boolean;
}

/**
 * Runs the Mock Accreditation Panel: three personas each critique the SSR
 * and produce likely site-visit questions with suggested answers. Without
 * an AI key it returns a clear "AI required" placeholder per persona
 * rather than fabricating a critique.
 */
export async function runMockPanel(runId: string): Promise<void> {
  const run = await prisma.mockPanelRun.findUniqueOrThrow({
    where: { id: runId },
    include: { document: true },
  });

  await prisma.mockPanelRun.update({
    where: { id: runId },
    data: { status: ReviewStatus.RUNNING },
  });

  try {
    const locale = run.document.language === "AR" ? "ar" : "en";

    // Broad retrieval across the whole SSR for panel context.
    const { chunks } = await hybridSearch({
      institutionId: run.institutionId,
      query:
        "mission goals management quality teaching learning students staff outcomes assessment evidence",
      documentIds: [run.documentId],
      limit: 10,
    });
    const contextChunks: PanelContextChunk[] = chunks.map((c, i) => ({
      index: i + 1,
      page: c.page,
      content: c.content,
    }));

    const results: PersonaResult[] = [];
    for (const persona of PERSONAS) {
      results.push(
        isAiEnabled()
          ? await runPersona(run, persona.key, contextChunks, locale)
          : {
              key: persona.key,
              nameEn: persona.nameEn,
              nameAr: persona.nameAr,
              critique:
                "AI simulation is not configured (set ANTHROPIC_API_KEY). The panel critique and site-visit questions require the Anthropic API.",
              strengths: [],
              concerns: [],
              questions: [],
              aiDisabled: true,
            }
      );
    }

    await prisma.mockPanelRun.update({
      where: { id: runId },
      data: {
        status: ReviewStatus.COMPLETED,
        resultsJson: { personas: results } as unknown as Prisma.InputJsonValue,
        completedAt: new Date(),
      },
    });
  } catch (err) {
    await prisma.mockPanelRun.update({
      where: { id: runId },
      data: { status: ReviewStatus.FAILED },
    });
    throw err;
  }
}

async function runPersona(
  run: { institutionId: string; documentId: string },
  personaKey: PersonaKey,
  chunks: PanelContextChunk[],
  locale: string
): Promise<PersonaResult> {
  const persona = PERSONAS.find((p) => p.key === personaKey)!;
  const system = buildSystemPrompt(personaKey, locale);
  const user = buildUserPrompt(chunks);

  const started = Date.now();
  const response = await getAnthropic().messages.create({
    model: AI_MODEL,
    max_tokens: 2000,
    system,
    messages: [{ role: "user", content: user }],
  });
  const raw = response.content
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n");

  await prisma.aiInteraction.create({
    data: {
      institutionId: run.institutionId,
      feature: "mock_panel",
      model: AI_MODEL,
      promptVersion: PROMPT_VERSION,
      inputHash: crypto.createHash("sha256").update(system + "\n" + user).digest("hex"),
      outputText: raw,
      tokensIn: response.usage.input_tokens,
      tokensOut: response.usage.output_tokens,
      latencyMs: Date.now() - started,
    },
  });

  const parsed = safeParse(raw);
  return {
    key: personaKey,
    nameEn: persona.nameEn,
    nameAr: persona.nameAr,
    critique: String(parsed?.critique ?? "").slice(0, 4000),
    strengths: toStringArray(parsed?.strengths),
    concerns: toStringArray(parsed?.concerns),
    questions: Array.isArray(parsed?.questions)
      ? parsed!.questions
          .slice(0, 8)
          .map((q) => ({
            question: String(q?.question ?? "").slice(0, 500),
            suggestedAnswer: String(q?.suggestedAnswer ?? "").slice(0, 1000),
          }))
          .filter((q) => q.question)
      : [],
  };
}

interface RawPanel {
  critique?: string;
  strengths?: unknown;
  concerns?: unknown;
  questions?: Array<{ question?: string; suggestedAnswer?: string }>;
}

function safeParse(raw: string): RawPanel | null {
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    return JSON.parse(raw.slice(start, end + 1)) as RawPanel;
  } catch {
    return null;
  }
}

function toStringArray(v: unknown): string[] {
  return Array.isArray(v) ? v.map((x) => String(x).slice(0, 500)).slice(0, 8) : [];
}
