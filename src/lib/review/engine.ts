import crypto from "node:crypto";

import { Prisma, ReviewStatus, Verdict } from "@prisma/client";

import { AI_MODEL, getAnthropic, isAiEnabled } from "@/lib/ai/client";
import {
  PROMPT_VERSION,
  buildSystemPrompt,
  buildUserPrompt,
  type ReviewContextChunk,
} from "@/lib/ai/prompts/reviewer";
import { isPlaceholder } from "@/lib/standards/schema";
import { prisma } from "@/lib/db";
import { hybridSearch, type RetrievedChunk } from "@/lib/rag/retrieval";

const VERDICT_WEIGHT: Record<Verdict, number | null> = {
  MET: 100,
  PARTIALLY_MET: 50,
  NOT_MET: 0,
  NOT_EVALUATED: null, // excluded from the readiness score
};

interface CriterionVerdict {
  criterionId: string;
  verdict: Verdict;
  score: number | null;
  findingText: string;
  citations: Array<{ documentId: string; chunkId: string; page: number | null; quote: string }>;
  recommendations: string[];
}

/**
 * Runs a rubric-based review of one document against one standards pack.
 * Per standard: retrieve passages from the review-subject document, then
 * either ask the AI to judge each criterion (JSON verdicts, cited) or —
 * when no AI key is configured — fall back to an evidence-coverage
 * heuristic that never claims compliance it cannot see.
 */
export async function runReview(reviewId: string): Promise<void> {
  const review = await prisma.review.findUniqueOrThrow({
    where: { id: reviewId },
    include: {
      document: true,
      pack: {
        include: {
          standards: {
            orderBy: { sortOrder: "asc" },
            include: { criteria: { orderBy: { sortOrder: "asc" } } },
          },
        },
      },
    },
  });

  await prisma.review.update({
    where: { id: reviewId },
    data: { status: ReviewStatus.RUNNING, error: null },
  });

  try {
    const locale = review.document.language === "AR" ? "ar" : "en";
    const allVerdicts: CriterionVerdict[] = [];

    for (const standard of review.pack.standards) {
      if (standard.criteria.length === 0) continue;

      // Retrieve document passages relevant to this standard.
      const query = [
        stripPlaceholder(standard.titleEn),
        ...standard.criteria.map((c) => stripPlaceholder(c.titleEn)),
      ]
        .filter(Boolean)
        .join(" ");

      const { chunks, confident } = await hybridSearch({
        institutionId: review.institutionId,
        query: query || standard.code,
        documentIds: [review.documentId],
        limit: 8,
      });

      const verdicts = isAiEnabled()
        ? await judgeWithAI(review, standard, chunks, locale)
        : judgeWithHeuristic(standard, chunks, confident);

      allVerdicts.push(...verdicts);
    }

    // Persist findings.
    await prisma.reviewFinding.deleteMany({ where: { reviewId } });
    if (allVerdicts.length > 0) {
      await prisma.reviewFinding.createMany({
        data: allVerdicts.map((v) => ({
          reviewId,
          criterionId: v.criterionId,
          verdict: v.verdict,
          score: v.score,
          findingText: v.findingText,
          citations: v.citations as unknown as Prisma.InputJsonValue,
          recommendations: v.recommendations as unknown as Prisma.InputJsonValue,
        })),
      });
    }

    const readinessScore = computeReadiness(allVerdicts);
    const summary = buildSummary(allVerdicts, readinessScore);

    await prisma.review.update({
      where: { id: reviewId },
      data: {
        status: ReviewStatus.COMPLETED,
        readinessScore,
        summary,
        completedAt: new Date(),
      },
    });

    // Record a readiness snapshot for the trendline (Phase 6).
    if (review.programId) {
      await prisma.readinessSnapshot.create({
        data: {
          institutionId: review.institutionId,
          programId: review.programId,
          packId: review.packId,
          score: readinessScore,
          source: `review:${reviewId}`,
        },
      });
    }
  } catch (err) {
    await prisma.review.update({
      where: { id: reviewId },
      data: {
        status: ReviewStatus.FAILED,
        error: (err instanceof Error ? err.message : String(err)).slice(0, 2000),
      },
    });
    throw err;
  }
}

function stripPlaceholder(text: string): string {
  return isPlaceholder(text) ? "" : text;
}

function computeReadiness(verdicts: CriterionVerdict[]): number {
  const scored = verdicts
    .map((v) => (v.score != null ? v.score : VERDICT_WEIGHT[v.verdict]))
    .filter((s): s is number => s != null);
  if (scored.length === 0) return 0;
  return Math.round(scored.reduce((a, b) => a + b, 0) / scored.length);
}

function buildSummary(verdicts: CriterionVerdict[], score: number): string {
  const counts = { MET: 0, PARTIALLY_MET: 0, NOT_MET: 0, NOT_EVALUATED: 0 };
  for (const v of verdicts) counts[v.verdict]++;
  return `Readiness ${score}/100 across ${verdicts.length} criteria — Met: ${counts.MET}, Partially Met: ${counts.PARTIALLY_MET}, Not Met: ${counts.NOT_MET}.`;
}

async function judgeWithAI(
  review: { id: string; institutionId: string; documentId: string },
  standard: {
    titleEn: string;
    code: string;
    criteria: Array<{ id: string; code: string; titleEn: string; descriptionEn: string | null }>;
  },
  chunks: RetrievedChunk[],
  locale: string
): Promise<CriterionVerdict[]> {
  const contextChunks: ReviewContextChunk[] = chunks.map((c, i) => ({
    index: i + 1,
    page: c.page,
    headingPath: c.headingPath,
    content: c.content,
  }));

  const system = buildSystemPrompt(locale);
  const user = buildUserPrompt({
    standardTitle: isPlaceholder(standard.titleEn) ? standard.code : standard.titleEn,
    criteria: standard.criteria.map((c) => ({
      id: c.id,
      code: c.code,
      title: isPlaceholder(c.titleEn) ? c.code : c.titleEn,
      description: isPlaceholder(c.descriptionEn) ? null : c.descriptionEn,
    })),
    chunks: contextChunks,
  });

  const started = Date.now();
  const response = await getAnthropic().messages.create({
    model: AI_MODEL,
    max_tokens: 3000,
    system,
    messages: [{ role: "user", content: user }],
  });
  const raw = response.content
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n");

  await prisma.aiInteraction.create({
    data: {
      institutionId: review.institutionId,
      feature: "reviewer",
      model: AI_MODEL,
      promptVersion: PROMPT_VERSION,
      inputHash: crypto.createHash("sha256").update(system + "\n" + user).digest("hex"),
      outputText: raw,
      tokensIn: response.usage.input_tokens,
      tokensOut: response.usage.output_tokens,
      latencyMs: Date.now() - started,
    },
  });

  const parsed = safeParseVerdicts(raw);
  const byId = new Map(standard.criteria.map((c) => [c.id, c]));

  return standard.criteria.map((criterion) => {
    const v = parsed.find((p) => p.criterionId === criterion.id);
    if (!v || !byId.has(criterion.id)) {
      return {
        criterionId: criterion.id,
        verdict: Verdict.NOT_EVALUATED,
        score: null,
        findingText: "The reviewer did not return a verdict for this criterion.",
        citations: [],
        recommendations: [],
      };
    }
    const citations = (v.citations ?? [])
      .map((idx) => chunks[idx - 1])
      .filter(Boolean)
      .map((c) => ({
        documentId: c.documentId,
        chunkId: c.chunkId,
        page: c.page,
        quote: c.content.slice(0, 300),
      }));
    return {
      criterionId: criterion.id,
      verdict: normalizeVerdict(v.verdict),
      score: typeof v.score === "number" ? clamp(v.score) : null,
      findingText: String(v.findingText ?? "").slice(0, 4000),
      citations,
      recommendations: Array.isArray(v.recommendations)
        ? v.recommendations.map((r) => String(r).slice(0, 500)).slice(0, 5)
        : [],
    };
  });
}

/**
 * Evidence-coverage heuristic used when no AI key is configured. It keys
 * off whether the standard's theme is actually covered in the document
 * (retrieval confidence for this standard) so standards the document
 * omits come back NOT_MET while covered ones come back PARTIALLY_MET.
 * It never claims full compliance ("Met") — that needs AI judgement.
 */
function judgeWithHeuristic(
  standard: {
    titleEn: string;
    criteria: Array<{ id: string; titleEn: string }>;
  },
  chunks: RetrievedChunk[],
  standardCovered: boolean
): CriterionVerdict[] {
  return standard.criteria.map((criterion) => {
    // Confidence (≥2 distinct standard-title terms matched in the
    // document) is the signal that this standard's theme is covered.
    const match = standardCovered ? chunks[0] : undefined;

    if (match) {
      return {
        criterionId: criterion.id,
        verdict: Verdict.PARTIALLY_MET,
        score: 50,
        findingText:
          "Evidence addressing this standard was located in the document, but a full compliance judgement requires AI review (set ANTHROPIC_API_KEY). Verify criterion-level coverage manually.",
        citations: [
          {
            documentId: match.documentId,
            chunkId: match.chunkId,
            page: match.page,
            quote: match.content.slice(0, 300),
          },
        ],
        recommendations: [
          "Enable AI review for a criterion-level compliance verdict and targeted improvements.",
        ],
      };
    }
    return {
      criterionId: criterion.id,
      verdict: Verdict.NOT_MET,
      score: 0,
      findingText:
        "No evidence addressing this standard was found in the document.",
      citations: [],
      recommendations: [
        "Add a section addressing this standard with supporting evidence and documentation.",
      ],
    };
  });
}

interface RawVerdict {
  criterionId: string;
  verdict: string;
  score?: number;
  findingText?: string;
  citations?: number[];
  recommendations?: string[];
}

function safeParseVerdicts(raw: string): RawVerdict[] {
  const start = raw.indexOf("[");
  const end = raw.lastIndexOf("]");
  if (start < 0 || end < 0 || end <= start) return [];
  try {
    const parsed = JSON.parse(raw.slice(start, end + 1));
    return Array.isArray(parsed) ? (parsed as RawVerdict[]) : [];
  } catch {
    return [];
  }
}

function normalizeVerdict(v: string): Verdict {
  const up = String(v).toUpperCase().replace(/\s+/g, "_");
  if (up === "MET") return Verdict.MET;
  if (up === "PARTIALLY_MET" || up === "PARTIAL") return Verdict.PARTIALLY_MET;
  if (up === "NOT_MET") return Verdict.NOT_MET;
  return Verdict.NOT_EVALUATED;
}

function clamp(n: number): number {
  return Math.max(0, Math.min(100, Math.round(n)));
}
