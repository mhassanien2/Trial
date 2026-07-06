import { notFound } from "next/navigation";

import { DocumentWizard } from "@/components/templates/document-wizard";
import { isAiEnabled } from "@/lib/ai/client";
import { prisma } from "@/lib/db";
import {
  generatedContentSchema,
  templateSchemaSchema,
} from "@/lib/templates/schema";
import { requireTenant } from "@/lib/tenant";

export default async function GeneratedDocumentPage({
  params,
}: {
  params: Promise<{ locale: string; id: string }>;
}) {
  const { locale, id } = await params;
  const tenant = await requireTenant(locale);

  const doc = await prisma.generatedDocument.findFirst({
    where: { id, institutionId: tenant.institutionId },
    include: {
      template: { select: { name: true, formCode: true, schemaJson: true } },
      program: { select: { code: true, nameEn: true, nameAr: true } },
    },
  });
  if (!doc) notFound();

  const schema = templateSchemaSchema.parse(doc.template.schemaJson);
  const content = generatedContentSchema.parse(doc.contentJson ?? {});

  return (
    <DocumentWizard
      docId={doc.id}
      title={doc.title}
      templateName={doc.template.name}
      formCode={doc.template.formCode}
      programLabel={`${doc.program.code} — ${locale === "ar" ? doc.program.nameAr : doc.program.nameEn}`}
      schema={schema}
      initialContent={content}
      aiEnabled={isAiEnabled()}
    />
  );
}
