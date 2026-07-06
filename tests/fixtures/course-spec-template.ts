import {
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

/**
 * Builds a realistic NCAAA-style Course Specification (TP-153) template
 * as a DOCX buffer: label/value tables, a CLO table with an empty
 * prototype row, and label-colon paragraphs. Used by parser round-trip
 * tests and by the demo seed.
 */
export async function buildCourseSpecTemplate(): Promise<Buffer> {
  const bold = (text: string) => new Paragraph({ children: [new TextRun({ text, bold: true })] });
  const plain = (text: string) => new Paragraph({ children: [new TextRun({ text })] });

  const labelValueRow = (label: string) =>
    new TableRow({
      children: [
        new TableCell({
          width: { size: 35, type: WidthType.PERCENTAGE },
          children: [bold(label)],
        }),
        new TableCell({
          width: { size: 65, type: WidthType.PERCENTAGE },
          children: [plain("")],
        }),
      ],
    });

  const headerCell = (text: string) => new TableCell({ children: [bold(text)] });
  const emptyCell = () => new TableCell({ children: [plain("")] });

  const doc = new Document({
    sections: [
      {
        children: [
          new Paragraph({
            heading: HeadingLevel.TITLE,
            children: [new TextRun({ text: "Course Specification", bold: true })],
          }),
          plain("Institution: Demo University"),
          plain("Form Code: TP-153"),

          new Paragraph({
            heading: HeadingLevel.HEADING_1,
            children: [new TextRun("A. Course Identification")],
          }),
          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: [
              labelValueRow("Course Title"),
              labelValueRow("Course Code"),
              labelValueRow("Credit Hours"),
              labelValueRow("Program Name"),
              labelValueRow("Level/Year"),
            ],
          }),

          new Paragraph({
            heading: HeadingLevel.HEADING_1,
            children: [new TextRun("B. Course Description")],
          }),
          plain("Course description as stated in the approved program specification:"),
          plain(""),

          new Paragraph({
            heading: HeadingLevel.HEADING_1,
            children: [new TextRun("C. Course Learning Outcomes")],
          }),
          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: [
              new TableRow({
                children: [
                  headerCell("CLO Code"),
                  headerCell("Course Learning Outcome"),
                  headerCell("Aligned PLO"),
                  headerCell("Assessment Methods"),
                ],
              }),
              new TableRow({
                children: [emptyCell(), emptyCell(), emptyCell(), emptyCell()],
              }),
            ],
          }),

          new Paragraph({
            heading: HeadingLevel.HEADING_1,
            children: [new TextRun("D. Teaching Strategies")],
          }),
          plain("Main teaching strategies:"),
          plain(""),
        ],
      },
    ],
  });

  return Buffer.from(await Packer.toBuffer(doc));
}
