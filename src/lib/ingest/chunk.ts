import type { ExtractedPage } from "./extract";

export interface Chunk {
  content: string;
  page: number;
  headingPath: string | null;
  criterionCode: string | null;
}

const TARGET_SIZE = 1000; // chars
const MAX_SIZE = 1800;
const OVERLAP_PARAGRAPHS = 1;

// Heading heuristics for accreditation documents (EN + AR):
// "Standard 3 …", "المعيار الثالث", "3. Teaching and Learning", "3-2 …"
const HEADING_PATTERNS = [
  /^(standard|criterion|section|chapter|appendix)\s+\d+/i,
  /^(المعيار|المحك|الفصل|الملحق|القسم)\s/,
  /^\d+(?:[.-]\d+)*[.)]?\s+\S/,
  /^[A-Z][A-Z\s&,]{8,}$/, // ALL-CAPS lines
];

// Criterion code like "3-2", "4.1.2" at the start of a block.
const CRITERION_CODE = /^(\d+(?:[.-]\d+)+)\b/;

function isHeading(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed || trimmed.length > 120) return false;
  return HEADING_PATTERNS.some((p) => p.test(trimmed));
}

/**
 * Heading-aware chunking: paragraphs are grouped up to ~TARGET_SIZE chars,
 * a new chunk always starts at a detected heading or criterion code (so a
 * criterion's text stays in one chunk), and consecutive chunks overlap by
 * one paragraph for context continuity.
 */
export function chunkPages(pages: ExtractedPage[]): Chunk[] {
  const chunks: Chunk[] = [];
  let headingPath: string | null = null;

  let current: string[] = [];
  let currentPage = 1;
  let currentHeading: string | null = null;
  let currentCriterion: string | null = null;

  const flush = () => {
    const content = current.join("\n").trim();
    if (content.length > 0) {
      chunks.push({
        content,
        page: currentPage,
        headingPath: currentHeading,
        criterionCode: currentCriterion,
      });
    }
    // Overlap: carry the tail paragraph(s) into the next chunk.
    current = current.slice(-OVERLAP_PARAGRAPHS);
  };

  for (const { page, text } of pages) {
    const paragraphs = text
      .split(/\n{2,}|\r\n{2,}|(?<=[.؟!:])\s{3,}/)
      .flatMap((p) => p.split("\n"))
      .map((p) => p.replace(/\s+/g, " ").trim())
      .filter(Boolean);

    for (const para of paragraphs) {
      const heading = isHeading(para);
      const criterionMatch = para.match(CRITERION_CODE);

      const size = current.join("\n").length;
      if (heading || criterionMatch || size >= TARGET_SIZE) {
        if (size > 0) flush();
        currentPage = page;
        if (heading) {
          headingPath = para.slice(0, 120);
          currentHeading = headingPath;
          currentCriterion = criterionMatch?.[1] ?? null;
        } else if (criterionMatch) {
          currentHeading = headingPath;
          currentCriterion = criterionMatch[1];
        } else {
          currentHeading = headingPath;
          // keep currentCriterion (continuation of the same block)
        }
      }

      current.push(para);

      // Hard cap: never exceed MAX_SIZE.
      if (current.join("\n").length >= MAX_SIZE) {
        flush();
        currentPage = page;
      }
    }
  }
  if (current.join("\n").trim().length > 0) flush();

  // Drop near-empty fragments produced by overlap.
  return chunks.filter((c) => c.content.length >= 40);
}
