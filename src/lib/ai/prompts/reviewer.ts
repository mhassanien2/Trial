/**
 * AI Reviewer prompt — rubric-based compliance analysis of a completed
 * quality document (SSR, course/program spec, report) against one
 * standard's criteria, using ONLY retrieved passages from that document.
 *
 * The model returns a strict JSON array (one entry per criterion) so the
 * engine can persist structured findings. Every verdict must cite the
 * passages it relied on; a criterion with no supporting evidence is
 * "Not Met" with an empty citation list — the model never invents
 * evidence or assumes compliance.
 */
export const PROMPT_VERSION = "reviewer@v1";

export interface ReviewCriterionInput {
  id: string;
  code: string;
  title: string;
  description?: string | null;
}

export interface ReviewContextChunk {
  index: number; // 1-based citation marker
  page: number | null;
  headingPath: string | null;
  content: string;
}

export function buildSystemPrompt(locale: string): string {
  const lang =
    locale === "ar"
      ? "Write findingText and recommendations in Arabic."
      : "Write findingText and recommendations in English.";
  return [
    "You are an accreditation reviewer assessing a completed quality document against a set of standard criteria.",
    "You are given numbered passages extracted from the document under review, and a list of criteria for ONE standard.",
    "",
    "For EACH criterion, decide a verdict:",
    '- "MET": the document clearly and fully addresses the criterion, evidenced by the passages.',
    '- "PARTIALLY_MET": the document addresses the criterion but with gaps, weak language, or missing evidence.',
    '- "NOT_MET": the document does not address the criterion, or no supporting passage exists.',
    "",
    "STRICT RULES:",
    "1. Base every judgement ONLY on the provided passages. Never assume compliance that the passages do not show. If no passage supports a criterion, it is NOT_MET with an empty citations array.",
    "2. Do not invent evidence, page numbers, or quotes.",
    "3. Cite the passage index numbers you relied on in the citations array.",
    "4. score is 0-100 for that criterion (MET≈85-100, PARTIALLY_MET≈40-70, NOT_MET≈0-30).",
    "5. Give 1-3 concrete, actionable recommendations for anything not fully MET; empty array if MET.",
    "6. Reply with ONLY a JSON array, no prose, matching exactly:",
    '[{"criterionId":"...","verdict":"MET|PARTIALLY_MET|NOT_MET","score":0,"findingText":"...","citations":[1,2],"recommendations":["..."]}]',
    lang,
  ].join("\n");
}

export function buildUserPrompt(input: {
  standardTitle: string;
  criteria: ReviewCriterionInput[];
  chunks: ReviewContextChunk[];
}): string {
  const criteriaList = input.criteria
    .map((c) => `- criterionId=${c.id} [${c.code}] ${c.title}${c.description ? ` — ${c.description}` : ""}`)
    .join("\n");

  const context =
    input.chunks.length > 0
      ? input.chunks
          .map(
            (c) =>
              `[${c.index}]${c.page != null ? ` (page ${c.page})` : ""}${c.headingPath ? ` [${c.headingPath}]` : ""}\n${c.content}`
          )
          .join("\n\n---\n\n")
      : "(no passages retrieved from the document for this standard)";

  return [
    `STANDARD: ${input.standardTitle}`,
    "",
    "CRITERIA TO ASSESS:",
    criteriaList,
    "",
    "DOCUMENT PASSAGES:",
    context,
  ].join("\n");
}
