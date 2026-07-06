import {
  AlignmentType,
  Document,
  HeadingLevel,
  Packer,
  Paragraph,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} from "docx";
import type { Verdict } from "@prisma/client";

interface ReportFinding {
  standardTitle: string;
  criterionCode: string;
  criterionTitle: string;
  verdict: Verdict;
  score: number | null;
  findingText: string;
  citations: Array<{ page: number | null; quote: string }>;
  recommendations: string[];
}

const VERDICT_LABEL: Record<Verdict, string> = {
  MET: "Met",
  PARTIALLY_MET: "Partially Met",
  NOT_MET: "Not Met",
  NOT_EVALUATED: "Not Evaluated",
};

/** Renders an exportable AI review report as DOCX. */
export async function exportReviewReport(input: {
  documentTitle: string;
  packName: string;
  programLabel: string | null;
  readinessScore: number;
  summary: string;
  findings: ReportFinding[];
  rtl?: boolean;
}): Promise<Buffer> {
  const rtl = Boolean(input.rtl);
  const align = rtl ? AlignmentType.RIGHT : AlignmentType.LEFT;

  const p = (text: string, opts?: { bold?: boolean; bullet?: boolean }) =>
    new Paragraph({
      bidirectional: rtl,
      alignment: align,
      bullet: opts?.bullet ? { level: 0 } : undefined,
      children: [new TextRun({ text, bold: opts?.bold, rightToLeft: rtl })],
    });

  const children: Array<Paragraph | Table> = [
    new Paragraph({
      heading: HeadingLevel.TITLE,
      alignment: align,
      children: [new TextRun({ text: "AI Readiness Review Report", rightToLeft: rtl })],
    }),
    p(`Document: ${input.documentTitle}`),
    p(`Standards pack: ${input.packName}`),
    ...(input.programLabel ? [p(`Program: ${input.programLabel}`)] : []),
    p(`Overall readiness score: ${input.readinessScore} / 100`, { bold: true }),
    p(input.summary),
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      alignment: align,
      children: [new TextRun({ text: "Findings by criterion", rightToLeft: rtl })],
    }),
  ];

  let lastStandard = "";
  for (const f of input.findings) {
    if (f.standardTitle !== lastStandard) {
      children.push(
        new Paragraph({
          heading: HeadingLevel.HEADING_2,
          alignment: align,
          children: [new TextRun({ text: f.standardTitle, rightToLeft: rtl })],
        })
      );
      lastStandard = f.standardTitle;
    }

    children.push(
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        rows: [
          new TableRow({
            children: [
              new TableCell({
                width: { size: 70, type: WidthType.PERCENTAGE },
                children: [p(`${f.criterionCode} — ${f.criterionTitle}`, { bold: true })],
              }),
              new TableCell({
                width: { size: 30, type: WidthType.PERCENTAGE },
                children: [
                  p(
                    `${VERDICT_LABEL[f.verdict]}${f.score != null ? ` (${f.score}/100)` : ""}`,
                    { bold: true }
                  ),
                ],
              }),
            ],
          }),
        ],
      })
    );

    children.push(p(f.findingText));

    if (f.citations.length > 0) {
      children.push(p("Citations:", { bold: true }));
      for (const c of f.citations) {
        children.push(
          p(`${c.page != null ? `p.${c.page}: ` : ""}“${c.quote}”`, { bullet: true })
        );
      }
    }
    if (f.recommendations.length > 0) {
      children.push(p("Recommendations:", { bold: true }));
      for (const r of f.recommendations) children.push(p(r, { bullet: true }));
    }
  }

  const doc = new Document({
    sections: [{ children }],
    styles: {
      default: { document: { run: { font: rtl ? "Arial" : "Calibri", size: 22 } } },
    },
  });
  return Buffer.from(await Packer.toBuffer(doc));
}
