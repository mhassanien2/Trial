import mammoth from "mammoth";

export interface ExtractedPage {
  page: number; // 1-based
  text: string;
}

export interface ExtractionResult {
  pages: ExtractedPage[];
  pageCount: number;
}

export class UnsupportedFileTypeError extends Error {}

const PDF_MIMES = new Set(["application/pdf"]);
const DOCX_MIMES = new Set([
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
]);

export async function extractText(
  buffer: Buffer,
  mimeType: string
): Promise<ExtractionResult> {
  if (PDF_MIMES.has(mimeType)) return extractPdf(buffer);
  if (DOCX_MIMES.has(mimeType)) return extractDocx(buffer);
  // XLSX extraction pending a dependency decision (not in the locked stack).
  throw new UnsupportedFileTypeError(`Unsupported mime type: ${mimeType}`);
}

async function extractPdf(buffer: Buffer): Promise<ExtractionResult> {
  const { PDFParse } = await import("pdf-parse");
  const parser = new PDFParse({ data: new Uint8Array(buffer) });
  try {
    const result = await parser.getText();
    const pages: ExtractedPage[] = result.pages.map((p) => ({
      page: p.num,
      text: p.text,
    }));
    return { pages, pageCount: result.total };
  } finally {
    await parser.destroy();
  }
}

async function extractDocx(buffer: Buffer): Promise<ExtractionResult> {
  const result = await mammoth.extractRawText({ buffer });
  // DOCX has no fixed pagination; treat the whole body as one "page".
  return { pages: [{ page: 1, text: result.value }], pageCount: 1 };
}
