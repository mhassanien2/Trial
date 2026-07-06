/**
 * AI-assisted template field filling (document generator wizard).
 * Suggestions draw ONLY on retrieved institutional context; when the
 * context is insufficient the model must return the NOT_ENOUGH_INFO
 * sentinel so the app can fall back to [REQUIRES INPUT] — the generator
 * never invents content for an official template.
 */
export const PROMPT_VERSION = "template-fill@v1";

export const NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO";

export interface FillContextChunk {
  index: number;
  documentTitle: string;
  page: number | null;
  content: string;
}

export function buildSystemPrompt(locale: string): string {
  const lang =
    locale === "ar"
      ? "Write the suggestion in Arabic (Modern Standard Arabic)."
      : "Write the suggestion in English.";
  return [
    "You draft content for ONE field of an official accreditation template (e.g. an NCAAA course or program specification).",
    "",
    "STRICT RULES:",
    "1. Base the draft ONLY on the provided context passages and program facts. Do not use outside knowledge about the institution or invent facts, numbers, or policies.",
    `2. If the context does not contain enough information to draft this field, reply with exactly: ${NOT_ENOUGH_INFO}`,
    "3. Output only the field content itself — no headings, no explanations, no citation markers (the audit trail stores sources separately).",
    "4. Match the register of formal accreditation documents; be specific and concise.",
    lang,
  ].join("\n");
}

export function buildUserPrompt(input: {
  sectionHeading: string;
  fieldLabel: string;
  program: { name: string; code: string; degreeLevel: string; department?: string | null };
  documentTitle: string;
  chunks: FillContextChunk[];
}): string {
  const context =
    input.chunks.length > 0
      ? input.chunks
          .map(
            (c) =>
              `[${c.index}] (${c.documentTitle}${c.page != null ? `, page ${c.page}` : ""})\n${c.content}`
          )
          .join("\n\n---\n\n")
      : "(no passages retrieved)";

  return [
    `TEMPLATE: ${input.documentTitle}`,
    `SECTION: ${input.sectionHeading}`,
    `FIELD TO DRAFT: ${input.fieldLabel}`,
    "",
    "PROGRAM FACTS:",
    `- Name: ${input.program.name}`,
    `- Code: ${input.program.code}`,
    `- Degree level: ${input.program.degreeLevel}`,
    input.program.department ? `- Department: ${input.program.department}` : "",
    "",
    "CONTEXT PASSAGES:",
    context,
  ]
    .filter(Boolean)
    .join("\n");
}
