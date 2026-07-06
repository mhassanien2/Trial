/**
 * Demo seed: 1 institution, 5 users (one per role), 2 programs
 * (PharmD, MBBCH). Standards pack seeding (NCAAA / NAQAAE structure
 * from /data/standards) is added in Phase 2.
 *
 * Run with: pnpm db:seed
 * Demo password for all users: Demo1234!
 */
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import {
  PrismaClient,
  Role,
  DegreeLevel,
  DocLanguage,
  DocumentKind,
  IngestStatus,
  PackOrigin,
  PdcaPhase,
  PlanItemStatus,
  ActionStatus,
} from "@prisma/client";
import bcrypt from "bcryptjs";

import { standardsPackFileSchema } from "../src/lib/standards/schema";
import { parseDocxTemplate } from "../src/lib/templates/parser";
import { buildCourseSpecTemplate } from "../tests/fixtures/course-spec-template";
import { buildSampleSSR } from "../tests/fixtures/sample-ssr";
import { extractText } from "../src/lib/ingest/extract";
import { chunkPages } from "../src/lib/ingest/chunk";
import { embedTexts } from "../src/lib/ai/embeddings";

const DOCX_MIME =
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document";

const prisma = new PrismaClient();

const DEMO_PASSWORD = "Demo1234!";

/**
 * Seed a demo NCAAA-style Course Specification (TP-153) template so the
 * generator is demo-ready: build the DOCX, store it, parse it into a
 * Template schema, then create one sample generated course spec.
 */
async function seedDemoTemplate(institutionId: string, programId: string) {
  const buffer = await buildCourseSpecTemplate();
  const docId = "doc_tpl_coursespec";
  const storageKey = `${institutionId}/${docId}/TP-153_Course_Specification.docx`;
  const uploadsDir = path.join(__dirname, "..", "uploads", path.dirname(storageKey));
  fs.mkdirSync(uploadsDir, { recursive: true });
  fs.writeFileSync(path.join(__dirname, "..", "uploads", storageKey), buffer);

  const sha256 = crypto.createHash("sha256").update(buffer).digest("hex");
  const document = await prisma.document.upsert({
    where: { id: docId },
    update: {},
    create: {
      id: docId,
      institutionId,
      kind: DocumentKind.TEMPLATE_SOURCE,
      title: "Course Specification (TP-153)",
      language: DocLanguage.EN,
      storageKey,
      mimeType:
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      sizeBytes: buffer.length,
      sha256,
      metadata: { formCode: "TP-153" },
      ingestStatus: IngestStatus.READY,
    },
  });

  const schema = await parseDocxTemplate(buffer, {
    title: "Course Specification (TP-153)",
    language: "EN",
  });

  const template = await prisma.template.upsert({
    where: { sourceDocumentId: document.id },
    update: { schemaJson: schema },
    create: {
      institutionId,
      sourceDocumentId: document.id,
      name: "Course Specification (TP-153)",
      formCode: "TP-153",
      language: DocLanguage.EN,
      schemaJson: schema,
    },
  });

  await prisma.generatedDocument.upsert({
    where: { id: "gen_demo_coursespec" },
    update: {},
    create: {
      id: "gen_demo_coursespec",
      institutionId,
      programId,
      templateId: template.id,
      title: "PHT-415 Clinical Pharmacokinetics — Course Specification",
      language: DocLanguage.EN,
    },
  });

  const fieldCount = schema.sections
    .flatMap((s) => s.blocks)
    .reduce(
      (n, b) =>
        n +
        (b.kind === "field" ? 1 : 0) +
        (b.kind === "table"
          ? b.rows.flatMap((r) => r.cells).filter((c) => c.field).length
          : 0),
      0
    );
  console.log(
    `  Template: ${template.name} (${schema.sections.length} sections, ${fieldCount} fields)`
  );
}

/**
 * Seed a completed PharmD SSR as a REVIEW_SUBJECT and run it through the
 * ingestion steps (extract → chunk → embed → pgvector) so the AI Reviewer
 * has a demo-ready, retrievable document out of the box.
 */
async function seedSampleSSR(institutionId: string) {
  const buffer = await buildSampleSSR();
  const docId = "doc_ssr_pharmd";
  const storageKey = `${institutionId}/${docId}/PharmD_Self_Study_Report.docx`;
  fs.mkdirSync(path.join(__dirname, "..", "uploads", path.dirname(storageKey)), {
    recursive: true,
  });
  fs.writeFileSync(path.join(__dirname, "..", "uploads", storageKey), buffer);

  const sha256 = crypto.createHash("sha256").update(buffer).digest("hex");
  const document = await prisma.document.upsert({
    where: { id: docId },
    update: {},
    create: {
      id: docId,
      institutionId,
      kind: DocumentKind.REVIEW_SUBJECT,
      title: "PharmD Self-Study Report (SSR)",
      language: DocLanguage.EN,
      storageKey,
      mimeType: DOCX_MIME,
      sizeBytes: buffer.length,
      sha256,
      ingestStatus: IngestStatus.PROCESSING,
    },
  });

  const { pages, pageCount } = await extractText(buffer, DOCX_MIME);
  const chunks = chunkPages(pages);
  await prisma.documentChunk.deleteMany({ where: { documentId: document.id } });

  const embeddings = await embedTexts(chunks.map((c) => c.content));
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    const vector = `[${embeddings[i].join(",")}]`;
    await prisma.$executeRawUnsafe(
      `INSERT INTO "DocumentChunk"
        (id, "documentId", "chunkIndex", content, embedding, page,
         "headingPath", "criterionCode", "tokenCount", metadata)
       VALUES ($1,$2,$3,$4,$5::vector,$6,$7,$8,$9,'{}'::jsonb)`,
      crypto.randomUUID(),
      document.id,
      i,
      c.content,
      vector,
      c.page,
      c.headingPath,
      c.criterionCode,
      Math.ceil(c.content.length / 4)
    );
  }

  await prisma.document.update({
    where: { id: document.id },
    data: { ingestStatus: IngestStatus.READY, pageCount },
  });

  console.log(`  SSR: ${document.title} (${chunks.length} chunks ingested)`);
}

/**
 * Seed a demo quality-cycle workspace for a program: PDCA plan items,
 * an evidence file linked to two NCAAA criteria, and improvement actions.
 */
async function seedProgramWorkspace(
  institutionId: string,
  programId: string,
  ownerId: string
) {
  const planItems: Array<[PdcaPhase, string, PlanItemStatus]> = [
    [PdcaPhase.PLAN, "Prepare Self-Study Report (SSR)", PlanItemStatus.IN_PROGRESS],
    [PdcaPhase.PLAN, "Update Program Specification (TP-151)", PlanItemStatus.PLANNED],
    [PdcaPhase.DO, "Draft course specifications for core courses", PlanItemStatus.IN_PROGRESS],
    [PdcaPhase.CHECK, "Run internal mock review against NCAAA", PlanItemStatus.PLANNED],
    [PdcaPhase.ACT, "Close gaps from last review cycle", PlanItemStatus.PLANNED],
  ];
  for (const [phase, title, status] of planItems) {
    const existing = await prisma.planItem.findFirst({
      where: { programId, title },
      select: { id: true },
    });
    if (!existing) {
      await prisma.planItem.create({
        data: { institutionId, programId, phase, title, status },
      });
    }
  }

  // Evidence document + links to two NCAAA program criteria.
  const evId = "doc_evidence_demo";
  const evKey = `${institutionId}/${evId}/Curriculum_Committee_Minutes.txt`;
  fs.mkdirSync(path.join(__dirname, "..", "uploads", path.dirname(evKey)), {
    recursive: true,
  });
  const evBuffer = Buffer.from(
    "Curriculum Committee Minutes — approval of PLO/CLO mapping and assessment plan.\n",
    "utf8"
  );
  fs.writeFileSync(path.join(__dirname, "..", "uploads", evKey), evBuffer);
  await prisma.document.upsert({
    where: { id: evId },
    update: {},
    create: {
      id: evId,
      institutionId,
      programId,
      kind: DocumentKind.EVIDENCE,
      title: "Curriculum Committee Minutes",
      language: DocLanguage.EN,
      storageKey: evKey,
      mimeType: "text/plain",
      sizeBytes: evBuffer.length,
      sha256: crypto.createHash("sha256").update(evBuffer).digest("hex"),
      ingestStatus: IngestStatus.NOT_APPLICABLE,
    },
  });

  const ncaaaProg = await prisma.standardsPack.findFirst({
    where: { code: "NCAAA-PROG-2022" },
    include: {
      standards: {
        orderBy: { sortOrder: "asc" },
        include: { criteria: { orderBy: { sortOrder: "asc" }, take: 2 } },
      },
    },
  });
  const teachingStd = ncaaaProg?.standards.find((s) => s.code === "S3");
  for (const criterion of teachingStd?.criteria ?? []) {
    await prisma.evidenceLink.upsert({
      where: {
        documentId_criterionId_programId: {
          documentId: evId,
          criterionId: criterion.id,
          programId,
        },
      },
      update: {},
      create: {
        institutionId,
        programId,
        documentId: evId,
        criterionId: criterion.id,
        note: "Approved mapping and assessment plan",
      },
    });
  }

  const actions: Array<[string, ActionStatus]> = [
    ["Document a formal risk register for the program", ActionStatus.OPEN],
    ["Record staff professional-development participation systematically", ActionStatus.IN_PROGRESS],
  ];
  for (const [title, status] of actions) {
    const existing = await prisma.improvementAction.findFirst({
      where: { programId, title },
      select: { id: true },
    });
    if (!existing) {
      await prisma.improvementAction.create({
        data: { institutionId, programId, title, status, ownerId },
      });
    }
  }

  console.log(
    `  Workspace: ${planItems.length} plan items, evidence linked to ${teachingStd?.criteria.length ?? 0} criteria, ${actions.length} actions`
  );
}

/** Load every pack JSON from /data/standards/{sa,eg} and upsert it. */
async function seedStandardsPacks() {
  const base = path.join(__dirname, "..", "data", "standards");
  const files = fs
    .readdirSync(base, { recursive: true, encoding: "utf8" })
    .filter((f) => f.endsWith(".json"))
    .map((f) => path.join(base, f));

  for (const file of files) {
    const parsed = standardsPackFileSchema.parse(
      JSON.parse(fs.readFileSync(file, "utf8"))
    );

    const pack = await prisma.standardsPack.upsert({
      where: { code_version: { code: parsed.code, version: parsed.version } },
      update: {
        nameEn: parsed.nameEn,
        nameAr: parsed.nameAr,
        description: parsed.description,
        country: parsed.country,
      },
      create: {
        code: parsed.code,
        version: parsed.version,
        country: parsed.country,
        nameEn: parsed.nameEn,
        nameAr: parsed.nameAr,
        description: parsed.description,
        origin: PackOrigin.OFFICIAL,
        institutionId: null, // official packs are global
      },
    });

    // Upsert standards/criteria in place (by code) so criterion IDs stay
    // stable across re-seeds — evidence links, review findings and mappings
    // that reference criteria are never broken by re-running the seed.
    for (const [i, std] of parsed.standards.entries()) {
      const standard = await prisma.standard.upsert({
        where: { packId_code: { packId: pack.id, code: std.code } },
        update: {
          titleEn: std.titleEn,
          titleAr: std.titleAr,
          descriptionEn: std.descriptionEn,
          descriptionAr: std.descriptionAr,
          sortOrder: i,
        },
        create: {
          packId: pack.id,
          code: std.code,
          titleEn: std.titleEn,
          titleAr: std.titleAr,
          descriptionEn: std.descriptionEn,
          descriptionAr: std.descriptionAr,
          sortOrder: i,
        },
      });

      for (const [j, c] of std.criteria.entries()) {
        const criterion = await prisma.criterion.upsert({
          where: { standardId_code: { standardId: standard.id, code: c.code } },
          update: {
            titleEn: c.titleEn,
            titleAr: c.titleAr,
            descriptionEn: c.descriptionEn,
            descriptionAr: c.descriptionAr,
            sortOrder: j,
          },
          create: {
            standardId: standard.id,
            code: c.code,
            titleEn: c.titleEn,
            titleAr: c.titleAr,
            descriptionEn: c.descriptionEn,
            descriptionAr: c.descriptionAr,
            sortOrder: j,
          },
        });

        // Indicators / evidence requirements have no external references,
        // so replace them wholesale to reflect the latest JSON.
        await prisma.indicator.deleteMany({ where: { criterionId: criterion.id } });
        await prisma.evidenceRequirement.deleteMany({
          where: { criterionId: criterion.id },
        });
        if (c.indicators.length > 0) {
          await prisma.indicator.createMany({
            data: c.indicators.map((ind, k) => ({
              criterionId: criterion.id,
              code: ind.code,
              textEn: ind.textEn,
              textAr: ind.textAr,
              sortOrder: k,
            })),
          });
        }
        if (c.evidenceRequirements.length > 0) {
          await prisma.evidenceRequirement.createMany({
            data: c.evidenceRequirements.map((e, k) => ({
              criterionId: criterion.id,
              textEn: e.textEn,
              textAr: e.textAr,
              sortOrder: k,
            })),
          });
        }
      }
    }
    console.log(`  Pack: ${parsed.code}@${parsed.version} (${parsed.standards.length} standards)`);
  }
}

async function main() {
  const passwordHash = await bcrypt.hash(DEMO_PASSWORD, 10);

  const institution = await prisma.institution.upsert({
    where: { id: "inst_demo" },
    update: {},
    create: {
      id: "inst_demo",
      nameEn: "Demo University",
      nameAr: "الجامعة التجريبية",
      country: "SA",
      settings: { defaultLocale: "en" },
    },
  });

  const users: Array<{ email: string; name: string; role: Role }> = [
    { email: "admin@demo.edu", name: "Amal Al-Admin", role: Role.ADMIN },
    { email: "qa@demo.edu", name: "Qasim Al-Quality", role: Role.QA_DIRECTOR },
    {
      email: "coordinator@demo.edu",
      name: "Chandra Coordinator",
      role: Role.PROGRAM_COORDINATOR,
    },
    { email: "faculty@demo.edu", name: "Fatima Faculty", role: Role.FACULTY },
    { email: "reviewer@demo.edu", name: "Rania Reviewer", role: Role.REVIEWER },
  ];

  for (const u of users) {
    await prisma.user.upsert({
      where: { email: u.email },
      update: { role: u.role, isActive: true },
      create: {
        email: u.email,
        name: u.name,
        role: u.role,
        passwordHash,
        institutionId: institution.id,
      },
    });
  }

  const pharmD = await prisma.program.upsert({
    where: {
      institutionId_code: { institutionId: institution.id, code: "PHARMD" },
    },
    update: {},
    create: {
      institutionId: institution.id,
      code: "PHARMD",
      nameEn: "Doctor of Pharmacy (PharmD)",
      nameAr: "دكتور صيدلة",
      degreeLevel: DegreeLevel.BACHELOR,
      department: "College of Pharmacy",
      nqfLevel: "SAQF L7",
    },
  });

  const mbbch = await prisma.program.upsert({
    where: {
      institutionId_code: { institutionId: institution.id, code: "MBBCH" },
    },
    update: {},
    create: {
      institutionId: institution.id,
      code: "MBBCH",
      nameEn: "Bachelor of Medicine and Surgery (MBBCH)",
      nameAr: "بكالوريوس الطب والجراحة",
      degreeLevel: DegreeLevel.BACHELOR,
      department: "College of Medicine",
      nqfLevel: "SAQF L7",
    },
  });

  const coordinator = await prisma.user.findUniqueOrThrow({
    where: { email: "coordinator@demo.edu" },
  });
  const faculty = await prisma.user.findUniqueOrThrow({
    where: { email: "faculty@demo.edu" },
  });

  for (const [programId, userId, role] of [
    [pharmD.id, coordinator.id, Role.PROGRAM_COORDINATOR],
    [pharmD.id, faculty.id, Role.FACULTY],
    [mbbch.id, coordinator.id, Role.PROGRAM_COORDINATOR],
  ] as const) {
    await prisma.programMember.upsert({
      where: { programId_userId: { programId, userId } },
      update: { roleInProgram: role },
      create: { programId, userId, roleInProgram: role },
    });
  }

  console.log("Standards packs:");
  await seedStandardsPacks();

  console.log("Demo template:");
  await seedDemoTemplate(institution.id, pharmD.id);

  console.log("Demo SSR (ingested for AI Reviewer):");
  await seedSampleSSR(institution.id);

  console.log("Program workspace (PDCA, evidence, actions):");
  await seedProgramWorkspace(institution.id, pharmD.id, coordinator.id);

  console.log("Seed complete:");
  console.log(`  Institution: ${institution.nameEn} (${institution.id})`);
  console.log(`  Users: ${users.map((u) => u.email).join(", ")}`);
  console.log(`  Password (all demo users): ${DEMO_PASSWORD}`);
  console.log(`  Programs: ${pharmD.code}, ${mbbch.code}`);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
