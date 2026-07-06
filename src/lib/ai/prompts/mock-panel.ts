/**
 * Mock Accreditation Panel — three reviewer personas each critique the
 * SSR from their vantage point and generate likely site-visit questions
 * with suggested answers, grounded ONLY in retrieved passages.
 */
export const PROMPT_VERSION = "mock-panel@v1";

export const PERSONAS = [
  {
    key: "standards_auditor",
    nameEn: "Standards Auditor",
    nameAr: "مدقق المعايير",
    briefEn:
      "You focus on compliance with accreditation standards, evidence sufficiency, traceability, and documentation gaps.",
  },
  {
    key: "curriculum_expert",
    nameEn: "Curriculum Expert",
    nameAr: "خبير المناهج",
    briefEn:
      "You focus on curriculum design, learning outcomes, CLO–PLO alignment, assessment validity, and teaching quality.",
  },
  {
    key: "student_experience_reviewer",
    nameEn: "Student-Experience Reviewer",
    nameAr: "مراجع تجربة الطالب",
    briefEn:
      "You focus on the student journey: admission, advising, support services, progression, satisfaction, and outcomes.",
  },
] as const;

export type PersonaKey = (typeof PERSONAS)[number]["key"];

export interface PanelContextChunk {
  index: number;
  page: number | null;
  content: string;
}

export function buildSystemPrompt(personaKey: PersonaKey, locale: string): string {
  const persona = PERSONAS.find((p) => p.key === personaKey)!;
  const lang =
    locale === "ar" ? "Write all output in Arabic." : "Write all output in English.";
  return [
    `You are a member of a mock accreditation site-visit panel, acting as the ${persona.nameEn}.`,
    persona.briefEn,
    "You are reviewing a Self-Study Report (SSR). You are given numbered passages from it.",
    "",
    "STRICT RULES:",
    "1. Base your critique and questions ONLY on the provided passages. Do not invent facts about the program.",
    "2. Be specific and constructive; reference what the SSR does or fails to show.",
    "3. Produce likely site-visit questions a real reviewer in your role would ask, each with a concise suggested answer the program could give (or a note that evidence appears missing).",
    "4. Reply with ONLY JSON matching exactly:",
    '{"critique":"...","strengths":["..."],"concerns":["..."],"questions":[{"question":"...","suggestedAnswer":"..."}]}',
    lang,
  ].join("\n");
}

export function buildUserPrompt(chunks: PanelContextChunk[]): string {
  const context =
    chunks.length > 0
      ? chunks
          .map((c) => `[${c.index}]${c.page != null ? ` (page ${c.page})` : ""}\n${c.content}`)
          .join("\n\n---\n\n")
      : "(no passages retrieved)";
  return `SSR PASSAGES:\n\n${context}`;
}
