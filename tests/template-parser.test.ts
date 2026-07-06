import assert from "node:assert/strict";
import { test } from "node:test";

import mammoth from "mammoth";

import { exportDocx } from "../src/lib/templates/export-docx";
import {
  htmlToNodes,
  isPlaceholderText,
  parseDocxTemplate,
  parseHtmlTemplate,
} from "../src/lib/templates/parser";
import {
  REQUIRES_INPUT,
  collectFields,
  templateSchemaSchema,
} from "../src/lib/templates/schema";
import { buildCourseSpecTemplate } from "./fixtures/course-spec-template";

// ───────────────────────── unit: heuristics ─────────────────────────

test("placeholder detection", () => {
  assert.equal(isPlaceholderText(""), true);
  assert.equal(isPlaceholderText("   "), true);
  assert.equal(isPlaceholderText("......"), true);
  assert.equal(isPlaceholderText("______"), true);
  assert.equal(isPlaceholderText("[Enter the course name]"), true);
  assert.equal(isPlaceholderText("Click here to enter text."), true);
  assert.equal(isPlaceholderText("اكتب هنا"), true);
  assert.equal(isPlaceholderText("Course Title"), false);
  assert.equal(isPlaceholderText("3 credit hours"), false);
});

test("html tokenizer handles headings, paragraphs, lists, tables", () => {
  const nodes = htmlToNodes(
    `<h1>Section One</h1><p><strong>Bold Head</strong></p><p>Text &amp; more</p>` +
      `<ol><li>Item 1</li><li>Item 2</li></ol>` +
      `<table><tr><th>H1</th><th>H2</th></tr><tr><td colspan="2">Wide</td></tr></table>`
  );
  assert.equal(nodes[0].type, "heading");
  assert.equal(nodes[0].text, "Section One");
  assert.equal(nodes[1].bold, true);
  assert.equal(nodes[2].text, "Text & more");
  assert.equal(nodes.filter((n) => n.type === "li").length, 2);
  const table = nodes.find((n) => n.type === "table");
  assert.ok(table?.rows);
  assert.equal(table.rows[0].cells[0].header, true);
  assert.equal(table.rows[1].cells[0].colSpan, 2);
});

// ───────────────────────── unit: parsing ─────────────────────────

test("parses sections, inline fields and label+placeholder fields", () => {
  const schema = parseHtmlTemplate(
    `<h1>General Information</h1>` +
      `<p>Course Code: ......</p>` +
      `<p>Course description:</p><p></p>` +
      `<p>This is fixed guidance text that must be preserved.</p>`,
    { title: "Test Template" }
  );

  templateSchemaSchema.parse(schema); // valid against zod

  assert.equal(schema.sections.length, 1);
  const blocks = schema.sections[0].blocks;
  const fields = blocks.filter((b) => b.kind === "field");
  const statics = blocks.filter((b) => b.kind === "static");

  assert.equal(fields.length, 2);
  assert.equal(fields[0].kind === "field" && fields[0].label, "Course Code");
  assert.equal(fields[1].kind === "field" && fields[1].label, "Course description");
  assert.equal(statics.length, 1);
  assert.match(statics[0].kind === "static" ? statics[0].text : "", /fixed guidance/);
});

test("parses label/value tables into row-labelled fields", () => {
  const schema = parseHtmlTemplate(
    `<h1>A. Identification</h1>` +
      `<table>` +
      `<tr><td><strong>Course Title</strong></td><td></td></tr>` +
      `<tr><td><strong>Credit Hours</strong></td><td></td></tr>` +
      `</table>`,
    { title: "T" }
  );
  const table = schema.sections[0].blocks.find((b) => b.kind === "table");
  assert.ok(table && table.kind === "table");
  const fieldCells = table.rows.flatMap((r) => r.cells).filter((c) => c.field);
  assert.equal(fieldCells.length, 2);
  assert.equal(fieldCells[0].field?.label, "Course Title");
  // "Credit Hours" should infer a short text input
  assert.equal(fieldCells[1].field?.fieldType, "text");
});

test("detects repeating prototype row under a header row", () => {
  const schema = parseHtmlTemplate(
    `<h1>CLOs</h1>` +
      `<table>` +
      `<tr><th>Code</th><th>Outcome</th></tr>` +
      `<tr><td></td><td></td></tr>` +
      `</table>`,
    { title: "T" }
  );
  const table = schema.sections[0].blocks.find((b) => b.kind === "table");
  assert.ok(table && table.kind === "table");
  assert.ok(table.repeatingRow, "repeating row prototype expected");
  assert.equal(table.repeatingRow?.cells.length, 2);
  // header row remains as the only fixed row
  assert.equal(table.rows.length, 1);
  assert.equal(table.rows[0].cells[0].header, true);
});

// ───────────────────── round-trip fidelity ─────────────────────

test("round-trip: DOCX template → schema → filled DOCX preserves structure", async () => {
  const templateBuffer = await buildCourseSpecTemplate();
  const schema = await parseDocxTemplate(templateBuffer, {
    title: "Course Specification",
  });

  // Section structure preserved, in order.
  const headings = schema.sections.map((s) => s.heading);
  assert.deepEqual(
    headings.filter((h) => /^[A-D]\./.test(h)),
    [
      "A. Course Identification",
      "B. Course Description",
      "C. Course Learning Outcomes",
      "D. Teaching Strategies",
    ]
  );

  const fields = collectFields(schema);
  const byLabel = (label: string) =>
    fields.find((f) => f.field.label === label)?.field;

  const courseTitle = byLabel("Course Title");
  const courseCode = byLabel("Course Code");
  assert.ok(courseTitle && courseCode, "identification fields parsed");

  const cloTable = schema.sections
    .flatMap((s) => s.blocks)
    .find((b) => b.kind === "table" && b.repeatingRow);
  assert.ok(cloTable && cloTable.kind === "table", "CLO table with prototype row");

  // Fill only some fields; export must mark the rest, not invent them.
  const filled = {
    [courseTitle.id]: "Clinical Pharmacokinetics",
    [courseCode.id]: "PHT-415",
    [cloTable.id]: {
      rows: [
        Object.fromEntries(
          cloTable.repeatingRow!.cells.map((c, i) => [
            c.field!.id,
            ["CLO1", "Apply PK principles to dosing", "PLO2", "Written exam"][i],
          ])
        ),
      ],
    },
  };

  const out = await exportDocx(schema, filled);
  const { value: text } = await mammoth.extractRawText({ buffer: out });

  // Structure: every section heading present, in order.
  let lastIndex = -1;
  for (const h of headings) {
    const idx = text.indexOf(h);
    assert.ok(idx >= 0, `heading missing: ${h}`);
    assert.ok(idx > lastIndex, `heading out of order: ${h}`);
    lastIndex = idx;
  }

  // Static template text preserved.
  assert.match(text, /approved program specification/);
  // Table headers preserved.
  for (const h of ["CLO Code", "Course Learning Outcome", "Aligned PLO", "Assessment Methods"]) {
    assert.ok(text.includes(h), `table header missing: ${h}`);
  }
  // Filled values present.
  assert.ok(text.includes("Clinical Pharmacokinetics"));
  assert.ok(text.includes("PHT-415"));
  assert.ok(text.includes("Apply PK principles to dosing"));
  // Unfilled fields are marked, never fabricated.
  assert.ok(text.includes(REQUIRES_INPUT), "unfilled fields must carry [REQUIRES INPUT]");
});
