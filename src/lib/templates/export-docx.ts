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

import {
  REQUIRES_INPUT,
  type GeneratedContent,
  type TableRow as SchemaRow,
  type TemplateSchema,
} from "./schema";

/**
 * Renders a filled document from the parsed template schema + content.
 * Template fidelity rules:
 *  - sections, headings, static text and tables are emitted in schema
 *    order, exactly once — the exporter cannot invent structure;
 *  - any unfilled field renders as the literal "[REQUIRES INPUT]"
 *    marker rather than fabricated content.
 */
export async function exportDocx(
  schema: TemplateSchema,
  content: GeneratedContent
): Promise<Buffer> {
  const rtl = schema.language === "AR";
  const children: Array<Paragraph | Table> = [];

  const HEADINGS = [
    HeadingLevel.HEADING_1,
    HeadingLevel.HEADING_2,
    HeadingLevel.HEADING_3,
    HeadingLevel.HEADING_4,
    HeadingLevel.HEADING_5,
    HeadingLevel.HEADING_6,
  ] as const;

  const para = (text: string, opts?: { bold?: boolean; bullet?: boolean }) =>
    new Paragraph({
      bidirectional: rtl,
      alignment: rtl ? AlignmentType.RIGHT : undefined,
      bullet: opts?.bullet ? { level: 0 } : undefined,
      children: [
        new TextRun({
          text,
          bold: opts?.bold,
          rightToLeft: rtl,
        }),
      ],
    });

  const fieldValue = (fieldId: string): string => {
    const v = content[fieldId];
    if (typeof v === "string" && v.trim().length > 0) return v;
    return REQUIRES_INPUT;
  };

  children.push(
    new Paragraph({
      heading: HeadingLevel.TITLE,
      bidirectional: rtl,
      alignment: rtl ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: [new TextRun({ text: schema.title, rightToLeft: rtl })],
    })
  );

  for (const section of schema.sections) {
    children.push(
      new Paragraph({
        heading: HEADINGS[Math.min(section.level, 6) - 1],
        bidirectional: rtl,
        alignment: rtl ? AlignmentType.RIGHT : undefined,
        children: [new TextRun({ text: section.heading, rightToLeft: rtl })],
      })
    );

    for (const block of section.blocks) {
      if (block.kind === "static") {
        children.push(para(block.text, { bold: block.bold, bullet: block.listLevel > 0 }));
      } else if (block.kind === "field") {
        children.push(para(`${block.label}: ${fieldValue(block.id)}`));
      } else {
        children.push(renderTable(block.rows, block.repeatingRow, block.id, content, para, fieldValue));
      }
    }
  }

  const doc = new Document({
    sections: [{ children }],
    styles: {
      default: {
        document: { run: { font: rtl ? "Arial" : "Calibri", size: 22 } },
      },
    },
  });

  return Buffer.from(await Packer.toBuffer(doc));
}

function renderTable(
  rows: SchemaRow[],
  repeatingRow: SchemaRow | null,
  tableId: string,
  content: GeneratedContent,
  para: (text: string, opts?: { bold?: boolean }) => Paragraph,
  fieldValue: (fieldId: string) => string
): Table {
  const renderRow = (row: SchemaRow, values?: Record<string, string>) =>
    new TableRow({
      children: row.cells.map(
        (cell) =>
          new TableCell({
            columnSpan: cell.colSpan > 1 ? cell.colSpan : undefined,
            children: [
              para(
                cell.field
                  ? values
                    ? values[cell.field.id]?.trim() || REQUIRES_INPUT
                    : fieldValue(cell.field.id)
                  : cell.static ?? "",
                { bold: cell.header }
              ),
            ],
          })
      ),
    });

  const tableRows = rows.map((r) => renderRow(r));

  // Extra rows entered by the user against the repeating prototype.
  if (repeatingRow) {
    const extra = content[tableId];
    if (extra && typeof extra === "object" && "rows" in extra) {
      for (const rowValues of extra.rows) {
        tableRows.push(renderRow(repeatingRow, rowValues));
      }
    }
  }

  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: tableRows,
  });
}
