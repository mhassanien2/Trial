import mammoth from "mammoth";

import {
  type FieldType,
  type TableRow,
  type TemplateBlock,
  type TemplateField,
  type TemplateSchema,
  type TemplateSection,
} from "./schema";

/**
 * DOCX template parser — the heart of the product.
 *
 * A DOCX template is converted (via mammoth) to predictable HTML, then
 * lifted into a TemplateSchema: sections (headings) → blocks (static
 * text, input fields, tables). Field detection is heuristic:
 *  - explicit placeholders:  [....], ……, "Click here to enter", ______
 *  - label paragraphs ending with ":" followed by an empty paragraph
 *  - empty table cells next to a label cell / under a header row
 *  - an entirely empty trailing table row ⇒ repeatable row prototype
 *
 * The same schema drives the wizard AND the exporter, so structure,
 * order, headings and tables are preserved by construction.
 */

// ─────────────────────── HTML mini-parser ───────────────────────
// mammoth emits a small, regular subset of HTML; this tokenizer only
// needs to understand that subset (no nested tables in templates).

interface HNode {
  type: "heading" | "para" | "li" | "table";
  level?: number;
  text?: string;
  bold?: boolean;
  rows?: Array<{ cells: Array<{ text: string; colSpan: number; header: boolean }> }>;
}

const BLOCK_RE = /<(h[1-6]|p|table|ol|ul)\b[^>]*>([\s\S]*?)<\/\1>/g;
const ROW_RE = /<tr\b[^>]*>([\s\S]*?)<\/tr>/g;
const CELL_RE = /<(td|th)\b([^>]*)>([\s\S]*?)<\/\1>/g;
const LI_RE = /<li\b[^>]*>([\s\S]*?)<\/li>/g;

export function decodeEntities(s: string): string {
  return s
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

function stripTags(s: string): string {
  return decodeEntities(s.replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ").trim();
}

function isFullyBold(inner: string): boolean {
  // "Fully bold" = has <strong> content and no visible text outside it.
  // Handles wrappers like <td><p><strong>…</strong></p></td>.
  if (!/<strong>/.test(inner)) return false;
  const outside = stripTags(inner.replace(/<strong>[\s\S]*?<\/strong>/g, " "));
  return outside.length === 0 && stripTags(inner).length > 0;
}

export function htmlToNodes(html: string): HNode[] {
  const nodes: HNode[] = [];
  let m: RegExpExecArray | null;
  BLOCK_RE.lastIndex = 0;
  while ((m = BLOCK_RE.exec(html))) {
    const [, tag, inner] = m;
    if (tag.startsWith("h")) {
      nodes.push({
        type: "heading",
        level: Number(tag[1]),
        text: stripTags(inner),
      });
    } else if (tag === "p") {
      const text = stripTags(inner);
      nodes.push({ type: "para", text, bold: isFullyBold(inner) });
    } else if (tag === "ol" || tag === "ul") {
      let li: RegExpExecArray | null;
      LI_RE.lastIndex = 0;
      while ((li = LI_RE.exec(inner))) {
        nodes.push({ type: "li", text: stripTags(li[1]) });
      }
    } else if (tag === "table") {
      const rows: NonNullable<HNode["rows"]> = [];
      let r: RegExpExecArray | null;
      ROW_RE.lastIndex = 0;
      while ((r = ROW_RE.exec(inner))) {
        const cells: Array<{ text: string; colSpan: number; header: boolean }> = [];
        let c: RegExpExecArray | null;
        CELL_RE.lastIndex = 0;
        while ((c = CELL_RE.exec(r[1]))) {
          const attrs = c[2];
          const colSpan = Number(/colspan="(\d+)"/.exec(attrs)?.[1] ?? 1);
          cells.push({
            text: stripTags(c[3]),
            colSpan,
            header: c[1] === "th" || isFullyBold(c[3]),
          });
        }
        if (cells.length > 0) rows.push({ cells });
      }
      nodes.push({ type: "table", rows });
    }
  }
  return nodes;
}

// ─────────────────────── field heuristics ───────────────────────

const PLACEHOLDER_PATTERNS = [
  /^[\s._…\-–—]*$/u, // empty / dots / underscores / dashes
  /^\[[^\]]*\]$/, // [Enter the course name]
  /click here to enter/i,
  /^(اكتب هنا|أدخل|يُعبأ|يعبأ هنا)/,
  /^\.{3,}$/,
];

export function isPlaceholderText(text: string): boolean {
  const t = text.trim();
  if (t.length === 0) return true;
  return PLACEHOLDER_PATTERNS.some((p) => p.test(t));
}

const DATE_HINT = /\b(date|تاريخ)\b/i;
const SHORT_HINT = /\b(code|number|no\.|credit|hours|رمز|رقم|عدد|ساعات)\b/i;

function inferFieldType(label: string): FieldType {
  if (DATE_HINT.test(label)) return "date";
  if (SHORT_HINT.test(label)) return "text";
  return "textarea";
}

function slugify(s: string): string {
  const ascii = s
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
  return ascii || "field";
}

// ─────────────────────── main parse ───────────────────────

export async function parseDocxTemplate(
  buffer: Buffer,
  opts: { title: string; language?: "EN" | "AR" }
): Promise<TemplateSchema> {
  const { value: html } = await mammoth.convertToHtml({ buffer });
  return parseHtmlTemplate(html, opts);
}

/** Split out for unit testing without a real DOCX. */
export function parseHtmlTemplate(
  html: string,
  opts: { title: string; language?: "EN" | "AR" }
): TemplateSchema {
  const nodes = htmlToNodes(html);
  const sections: TemplateSection[] = [];
  const usedIds = new Set<string>();

  const uniqueId = (base: string): string => {
    let id = base;
    let n = 2;
    while (usedIds.has(id)) id = `${base}-${n++}`;
    usedIds.add(id);
    return id;
  };

  let current: TemplateSection | null = null;
  const ensureSection = (heading: string, level: number): TemplateSection => {
    const section: TemplateSection = {
      id: uniqueId(`s-${slugify(heading)}`),
      order: sections.length,
      heading,
      level,
      blocks: [],
    };
    sections.push(section);
    return section;
  };

  const makeField = (
    label: string,
    placeholder: string | undefined
  ): TemplateField => ({
    kind: "field",
    id: uniqueId(`f-${slugify(label)}`),
    label,
    fieldType: inferFieldType(label),
    required: false,
    placeholder,
  });

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];

    // Section boundaries: real headings, or short fully-bold paragraphs.
    if (
      node.type === "heading" ||
      (node.type === "para" && node.bold && (node.text ?? "").length > 0 && (node.text ?? "").length <= 90)
    ) {
      current = ensureSection(node.text ?? "", node.level ?? 2);
      continue;
    }

    current ??= ensureSection(opts.title, 1);

    if (node.type === "para" || node.type === "li") {
      const text = node.text ?? "";

      // "Label: value-placeholder" on one line
      const inline = /^(.{2,80}?)[:：]\s*(.*)$/.exec(text);
      if (inline && isPlaceholderText(inline[2])) {
        current.blocks.push(makeField(inline[1].trim(), inline[2].trim() || undefined));
        continue;
      }

      // Label paragraph followed by an empty/placeholder paragraph
      const next = nodes[i + 1];
      if (
        /[:：]\s*$/.test(text) &&
        next?.type === "para" &&
        isPlaceholderText(next.text ?? "")
      ) {
        current.blocks.push(makeField(text.replace(/[:：]\s*$/, "").trim(), next.text?.trim() || undefined));
        i++; // consume the placeholder paragraph
        continue;
      }

      if (text.trim().length === 0) continue; // skip stray empties

      current.blocks.push({
        kind: "static",
        id: uniqueId("st"),
        text,
        bold: Boolean(node.bold),
        listLevel: node.type === "li" ? 1 : 0,
      });
      continue;
    }

    if (node.type === "table" && node.rows && node.rows.length > 0) {
      current.blocks.push(parseTable(node.rows, uniqueId, makeField));
    }
  }

  return {
    version: 1,
    title: opts.title,
    language: opts.language ?? "EN",
    sections,
  };
}

function parseTable(
  htmlRows: NonNullable<HNode["rows"]>,
  uniqueId: (base: string) => string,
  makeField: (label: string, placeholder: string | undefined) => TemplateField
): TemplateBlock {
  const headerRow = htmlRows[0]?.cells.every((c) => c.header && c.text.length > 0)
    ? htmlRows[0]
    : null;
  const columnLabels = headerRow?.cells.map((c) => c.text) ?? [];

  const bodyRows = headerRow ? htmlRows.slice(1) : htmlRows;
  const rows: TableRow[] = [];

  if (headerRow) {
    rows.push({
      cells: headerRow.cells.map((c) => ({
        static: c.text,
        colSpan: c.colSpan,
        header: true,
      })),
    });
  }

  let repeatingRow: TableRow | null = null;

  bodyRows.forEach((row, rowIdx) => {
    const rowLabel = !isPlaceholderText(row.cells[0]?.text ?? "")
      ? row.cells[0].text
      : null;

    const isLastRow = rowIdx === bodyRows.length - 1;
    const allEmpty = row.cells.every((c) => isPlaceholderText(c.text));

    const cells = row.cells.map((cell, colIdx) => {
      const colLabel = columnLabels[colIdx];
      if (isPlaceholderText(cell.text) || (rowLabel && colIdx > 0 && isPlaceholderText(cell.text))) {
        const label =
          rowLabel && colIdx > 0
            ? colLabel
              ? `${rowLabel} — ${colLabel}`
              : rowLabel
            : colLabel ?? `Row ${rowIdx + 1} / Col ${colIdx + 1}`;
        const f = makeField(label, cell.text.trim() || undefined);
        return {
          field: {
            id: f.id,
            label: f.label,
            fieldType: f.fieldType,
            required: f.required,
            placeholder: f.placeholder,
          },
          colSpan: cell.colSpan,
          header: false,
        };
      }
      return { static: cell.text, colSpan: cell.colSpan, header: cell.header };
    });

    // A fully-empty last body row (with a header) is a repeatable prototype.
    if (allEmpty && isLastRow && columnLabels.length > 0) {
      repeatingRow = { cells };
    } else {
      rows.push({ cells });
    }
  });

  return { kind: "table", id: uniqueId("tbl"), rows, repeatingRow };
}
