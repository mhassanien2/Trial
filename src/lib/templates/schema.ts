import { z } from "zod";

/**
 * Template schema: the parsed representation of an uploaded DOCX template.
 * It drives BOTH the guided input wizard and the DOCX/PDF renderer, which
 * is what guarantees template fidelity — the exporter can only emit what
 * the parser captured, in the captured order.
 */

export const fieldTypeSchema = z.enum(["text", "textarea", "date", "number"]);
export type FieldType = z.infer<typeof fieldTypeSchema>;

export const templateFieldSchema = z.object({
  kind: z.literal("field"),
  id: z.string(),
  label: z.string(),
  fieldType: fieldTypeSchema,
  required: z.boolean().default(false),
  /** original placeholder text found in the template, if any */
  placeholder: z.string().optional(),
});
export type TemplateField = z.infer<typeof templateFieldSchema>;

export const staticBlockSchema = z.object({
  kind: z.literal("static"),
  id: z.string(),
  /** plain text with minimal inline markers preserved (** for bold) */
  text: z.string(),
  bold: z.boolean().default(false),
  listLevel: z.number().int().min(0).default(0),
});
export type StaticBlock = z.infer<typeof staticBlockSchema>;

export const tableCellSchema = z.object({
  /** exactly one of static / field */
  static: z.string().optional(),
  field: templateFieldSchema.omit({ kind: true }).optional(),
  colSpan: z.number().int().min(1).default(1),
  header: z.boolean().default(false),
});
export type TableCell = z.infer<typeof tableCellSchema>;

export const tableRowSchema = z.object({
  cells: z.array(tableCellSchema),
});
export type TableRow = z.infer<typeof tableRowSchema>;

export const templateTableSchema = z.object({
  kind: z.literal("table"),
  id: z.string(),
  rows: z.array(tableRowSchema),
  /**
   * When the template's last body row is entirely empty it is treated as
   * a row prototype: users may add any number of rows shaped like it.
   */
  repeatingRow: tableRowSchema.nullable().default(null),
});
export type TemplateTable = z.infer<typeof templateTableSchema>;

export const templateBlockSchema = z.discriminatedUnion("kind", [
  staticBlockSchema,
  templateFieldSchema,
  templateTableSchema,
]);
export type TemplateBlock = z.infer<typeof templateBlockSchema>;

export const templateSectionSchema = z.object({
  id: z.string(),
  order: z.number().int(),
  heading: z.string(),
  level: z.number().int().min(1).max(6),
  blocks: z.array(templateBlockSchema),
});
export type TemplateSection = z.infer<typeof templateSectionSchema>;

export const templateSchemaSchema = z.object({
  version: z.literal(1),
  title: z.string(),
  language: z.enum(["EN", "AR"]).default("EN"),
  sections: z.array(templateSectionSchema),
});
export type TemplateSchema = z.infer<typeof templateSchemaSchema>;

/** Content for a generated document: field values + table row data. */
export const generatedContentSchema = z.record(
  z.string(),
  z.union([
    z.string(),
    z.object({
      /** extra rows for tables with a repeatingRow prototype */
      rows: z.array(z.record(z.string(), z.string())),
    }),
  ])
);
export type GeneratedContent = z.infer<typeof generatedContentSchema>;

export const REQUIRES_INPUT = "[REQUIRES INPUT]";

/** Collect all fields (top-level and inside tables) for wizard/AI use. */
export function collectFields(
  schema: TemplateSchema
): Array<{ sectionId: string; sectionHeading: string; field: Omit<TemplateField, "kind"> }> {
  const out: Array<{
    sectionId: string;
    sectionHeading: string;
    field: Omit<TemplateField, "kind">;
  }> = [];
  for (const section of schema.sections) {
    for (const block of section.blocks) {
      if (block.kind === "field") {
        out.push({ sectionId: section.id, sectionHeading: section.heading, field: block });
      } else if (block.kind === "table") {
        for (const row of block.rows) {
          for (const cell of row.cells) {
            if (cell.field) {
              out.push({
                sectionId: section.id,
                sectionHeading: section.heading,
                field: cell.field,
              });
            }
          }
        }
      }
    }
  }
  return out;
}
