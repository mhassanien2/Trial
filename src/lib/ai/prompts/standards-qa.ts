/**
 * Standards Q&A prompt (Standards Explorer cited chat).
 * The model may ONLY use the retrieved context; every claim must cite a
 * source marker. If the context is insufficient it must say so.
 */
export const PROMPT_VERSION = "standards-qa@v1";

export interface QaContextChunk {
  index: number; // 1-based citation marker
  documentTitle: string;
  page: number | null;
  headingPath: string | null;
  content: string;
}

export function buildSystemPrompt(locale: string): string {
  const langLine =
    locale === "ar"
      ? "Answer in Arabic (Modern Standard Arabic)."
      : "Answer in English.";

  return [
    "You are AccreditGenius, a quality-assurance and accreditation assistant for higher education institutions.",
    "You answer questions about accreditation standards and quality documents.",
    "",
    "STRICT RULES:",
    "1. Use ONLY the numbered context passages provided. Never use outside knowledge about standards, and never invent standards text, criterion numbers, or requirements.",
    "2. Every factual sentence must end with the citation marker(s) of the passage(s) that support it, in the form [1] or [1][3].",
    "3. If the context does not contain enough information to answer, reply exactly that the available documents do not cover the question, and suggest what document to upload. Do not guess.",
    "4. Quote exact wording when the user asks what a standard/criterion says.",
    langLine,
  ].join("\n");
}

export function buildUserPrompt(
  question: string,
  chunks: QaContextChunk[]
): string {
  const context = chunks
    .map((c) => {
      const loc = [
        c.documentTitle,
        c.page != null ? `page ${c.page}` : null,
        c.headingPath ? `section: ${c.headingPath}` : null,
      ]
        .filter(Boolean)
        .join(", ");
      return `[${c.index}] (${loc})\n${c.content}`;
    })
    .join("\n\n---\n\n");

  return `CONTEXT PASSAGES:\n\n${context}\n\nQUESTION: ${question}`;
}
