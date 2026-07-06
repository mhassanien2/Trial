import { NextResponse } from "next/server";

import { logAudit } from "@/lib/audit";
import { prisma } from "@/lib/db";
import { ForbiddenError } from "@/lib/rbac";
import { getStorage } from "@/lib/storage";
import { exportDocx } from "@/lib/templates/export-docx";
import {
  generatedContentSchema,
  templateSchemaSchema,
} from "@/lib/templates/schema";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";

/** GET → DOCX download rendered from the template schema + content. */
export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const tenant = await requireTenantWith("documents.view");

    const doc = await prisma.generatedDocument.findFirst({
      where: { id, institutionId: tenant.institutionId },
      include: { template: true },
    });
    if (!doc) return NextResponse.json({ error: "Not found" }, { status: 404 });

    const schema = templateSchemaSchema.parse(doc.template.schemaJson);
    const content = generatedContentSchema.parse(doc.contentJson ?? {});
    // Export in the generated document's language (title stays from template).
    schema.language = doc.language === "AR" ? "AR" : "EN";

    const buffer = await exportDocx(schema, content);

    // Keep a copy in storage for the audit trail.
    const exportKey = `${tenant.institutionId}/exports/${doc.id}.docx`;
    await getStorage().put(
      exportKey,
      buffer,
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    );
    await prisma.generatedDocument.update({
      where: { id: doc.id },
      data: { exportDocxKey: exportKey },
    });
    await logAudit({
      institutionId: tenant.institutionId,
      userId: tenant.userId,
      action: "generated.export.docx",
      entityType: "GeneratedDocument",
      entityId: doc.id,
    });

    const filename = `${doc.title.replace(/[^\w.-]+/g, "_")}.docx`;
    return new NextResponse(new Uint8Array(buffer), {
      headers: {
        "Content-Type":
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "Content-Disposition": `attachment; filename="${filename}"`,
      },
    });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[generated:export]", err);
    return NextResponse.json({ error: "Export failed" }, { status: 500 });
  }
}
