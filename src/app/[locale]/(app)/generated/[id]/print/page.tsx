import { notFound } from "next/navigation";

import { isRtl } from "@/i18n/routing";
import { prisma } from "@/lib/db";
import {
  REQUIRES_INPUT,
  generatedContentSchema,
  templateSchemaSchema,
  type GeneratedContent,
} from "@/lib/templates/schema";
import { requireTenant } from "@/lib/tenant";
import { PrintButton } from "@/components/templates/print-button";

/**
 * Print-optimized view of a generated document. The browser's
 * "Save as PDF" produces a PDF that mirrors the template structure
 * (same sections, tables, [REQUIRES INPUT] markers) — no extra
 * PDF dependency, and layout stays consistent with the DOCX export.
 */
export default async function PrintPage({
  params,
}: {
  params: Promise<{ locale: string; id: string }>;
}) {
  const { locale, id } = await params;
  const tenant = await requireTenant(locale);
  const rtl = isRtl(locale);

  const doc = await prisma.generatedDocument.findFirst({
    where: { id, institutionId: tenant.institutionId },
    include: { template: { select: { name: true, schemaJson: true } } },
  });
  if (!doc) notFound();

  const schema = templateSchemaSchema.parse(doc.template.schemaJson);
  const content: GeneratedContent = generatedContentSchema.parse(doc.contentJson ?? {});

  const val = (fieldId: string): string => {
    const v = content[fieldId];
    return typeof v === "string" && v.trim() ? v : REQUIRES_INPUT;
  };
  const extraRows = (tableId: string) => {
    const v = content[tableId];
    return v && typeof v === "object" && "rows" in v ? v.rows : [];
  };
  const mark = (s: string) =>
    s === REQUIRES_INPUT ? <span className="requires">{s}</span> : s;

  return (
    <div className="print-root" dir={rtl ? "rtl" : "ltr"}>
      <div className="no-print mb-4 flex justify-end">
        <PrintButton />
      </div>

      <h1 className="doc-title">{doc.title}</h1>

      {schema.sections.map((section) => (
        <section key={section.id} className="doc-section">
          <h2>{section.heading}</h2>
          {section.blocks.map((block) => {
            if (block.kind === "static") {
              return (
                <p key={block.id} className={block.bold ? "bold" : undefined}>
                  {block.text}
                </p>
              );
            }
            if (block.kind === "field") {
              return (
                <p key={block.id}>
                  <strong>{block.label}:</strong> {mark(val(block.id))}
                </p>
              );
            }
            return (
              <table key={block.id} className="doc-table">
                <tbody>
                  {block.rows.map((row, ri) => (
                    <tr key={ri}>
                      {row.cells.map((cell, ci) => (
                        <td key={ci} colSpan={cell.colSpan > 1 ? cell.colSpan : undefined}>
                          {cell.field
                            ? mark(val(cell.field.id))
                            : cell.header
                              ? <strong>{cell.static}</strong>
                              : cell.static}
                        </td>
                      ))}
                    </tr>
                  ))}
                  {block.repeatingRow
                    ? extraRows(block.id).map((rv, ri) => (
                        <tr key={`x-${ri}`}>
                          {block.repeatingRow!.cells.map((cell, ci) => (
                            <td key={ci}>
                              {cell.field
                                ? mark(rv[cell.field.id]?.trim() || REQUIRES_INPUT)
                                : cell.static}
                            </td>
                          ))}
                        </tr>
                      ))
                    : null}
                </tbody>
              </table>
            );
          })}
        </section>
      ))}
    </div>
  );
}
