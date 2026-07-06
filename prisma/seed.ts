/**
 * Demo seed: 1 institution, 5 users (one per role), 2 programs
 * (PharmD, MBBCH). Standards pack seeding (NCAAA / NAQAAE structure
 * from /data/standards) is added in Phase 2.
 *
 * Run with: pnpm db:seed
 * Demo password for all users: Demo1234!
 */
import fs from "node:fs";
import path from "node:path";

import { PrismaClient, Role, DegreeLevel, PackOrigin } from "@prisma/client";
import bcrypt from "bcryptjs";

import { standardsPackFileSchema } from "../src/lib/standards/schema";

const prisma = new PrismaClient();

const DEMO_PASSWORD = "Demo1234!";

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

    // Replace the structure wholesale so re-seeding stays idempotent.
    await prisma.standard.deleteMany({ where: { packId: pack.id } });
    for (const [i, std] of parsed.standards.entries()) {
      await prisma.standard.create({
        data: {
          packId: pack.id,
          code: std.code,
          titleEn: std.titleEn,
          titleAr: std.titleAr,
          descriptionEn: std.descriptionEn,
          descriptionAr: std.descriptionAr,
          sortOrder: i,
          criteria: {
            create: std.criteria.map((c, j) => ({
              code: c.code,
              titleEn: c.titleEn,
              titleAr: c.titleAr,
              descriptionEn: c.descriptionEn,
              descriptionAr: c.descriptionAr,
              sortOrder: j,
              indicators: {
                create: c.indicators.map((ind, k) => ({
                  code: ind.code,
                  textEn: ind.textEn,
                  textAr: ind.textAr,
                  sortOrder: k,
                })),
              },
              evidenceRequirements: {
                create: c.evidenceRequirements.map((e, k) => ({
                  textEn: e.textEn,
                  textAr: e.textAr,
                  sortOrder: k,
                })),
              },
            })),
          },
        },
      });
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
