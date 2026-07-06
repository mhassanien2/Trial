import { embedTexts } from "@/lib/ai/embeddings";
import { prisma } from "@/lib/db";
import { isPlaceholder } from "@/lib/standards/schema";

export interface MappedPair {
  from: { criterionId: string; standardCode: string; code: string; title: string };
  to: { criterionId: string; standardCode: string; code: string; title: string };
  overlapScore: number;
  fromHasEvidence: boolean;
  toHasEvidence: boolean;
}

export interface MapperResult {
  fromPack: string;
  toPack: string;
  pairs: MappedPair[];
  fromCovered: number;
  toCovered: number;
  bothCovered: number;
}

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // embeddings are L2-normalized (local) or near-unit; ranking only
}

/**
 * Cross-Standard Mapper: for a program, map each criterion of pack A to
 * its most-similar criterion in pack B (by embedded title+description),
 * and annotate whether the program has evidence on each side. The matrix
 * surfaces overlap (evidence on both) and gaps (evidence on one only).
 *
 * Mapping quality scales with official text; with placeholder packs the
 * structure is still produced but similarity is uninformative.
 */
export async function computeMapping(opts: {
  institutionId: string;
  programId: string;
  fromPackId: string;
  toPackId: string;
  locale: string;
}): Promise<MapperResult> {
  const [fromPack, toPack] = await Promise.all([
    loadPack(opts.institutionId, opts.fromPackId),
    loadPack(opts.institutionId, opts.toPackId),
  ]);
  if (!fromPack || !toPack) throw new Error("Unknown pack");

  const fromCriteria = flattenCriteria(fromPack, opts.locale);
  const toCriteria = flattenCriteria(toPack, opts.locale);

  // Evidence coverage per criterion for this program.
  const evidence = await prisma.evidenceLink.findMany({
    where: {
      programId: opts.programId,
      criterion: {
        standard: { packId: { in: [opts.fromPackId, opts.toPackId] } },
      },
    },
    select: { criterionId: true },
  });
  const covered = new Set(evidence.map((e) => e.criterionId));

  // Embed both sides; map each "from" to the best "to".
  const fromVecs = await embedTexts(fromCriteria.map((c) => c.text));
  const toVecs = await embedTexts(toCriteria.map((c) => c.text));

  const pairs: MappedPair[] = fromCriteria.map((fc, i) => {
    let bestJ = 0;
    let bestScore = -Infinity;
    for (let j = 0; j < toCriteria.length; j++) {
      const s = cosine(fromVecs[i], toVecs[j]);
      if (s > bestScore) {
        bestScore = s;
        bestJ = j;
      }
    }
    const tc = toCriteria[bestJ];
    return {
      from: {
        criterionId: fc.criterionId,
        standardCode: fc.standardCode,
        code: fc.code,
        title: fc.title,
      },
      to: {
        criterionId: tc.criterionId,
        standardCode: tc.standardCode,
        code: tc.code,
        title: tc.title,
      },
      overlapScore: Math.max(0, Math.min(1, bestScore)),
      fromHasEvidence: covered.has(fc.criterionId),
      toHasEvidence: covered.has(tc.criterionId),
    };
  });

  // Persist mappings for reuse (CriterionMapping is unique per from/to).
  for (const p of pairs) {
    await prisma.criterionMapping.upsert({
      where: {
        institutionId_fromCriterionId_toCriterionId: {
          institutionId: opts.institutionId,
          fromCriterionId: p.from.criterionId,
          toCriterionId: p.to.criterionId,
        },
      },
      update: { overlapScore: p.overlapScore, programId: opts.programId },
      create: {
        institutionId: opts.institutionId,
        programId: opts.programId,
        fromCriterionId: p.from.criterionId,
        toCriterionId: p.to.criterionId,
        overlapScore: p.overlapScore,
      },
    });
  }

  return {
    fromPack: opts.locale === "ar" && fromPack.nameAr ? fromPack.nameAr : fromPack.nameEn,
    toPack: opts.locale === "ar" && toPack.nameAr ? toPack.nameAr : toPack.nameEn,
    pairs,
    fromCovered: pairs.filter((p) => p.fromHasEvidence).length,
    toCovered: pairs.filter((p) => p.toHasEvidence).length,
    bothCovered: pairs.filter((p) => p.fromHasEvidence && p.toHasEvidence).length,
  };
}

async function loadPack(institutionId: string, packId: string) {
  return prisma.standardsPack.findFirst({
    where: {
      id: packId,
      OR: [{ institutionId: null }, { institutionId }],
    },
    include: {
      standards: {
        orderBy: { sortOrder: "asc" },
        include: { criteria: { orderBy: { sortOrder: "asc" } } },
      },
    },
  });
}

type LoadedPack = NonNullable<Awaited<ReturnType<typeof loadPack>>>;

function flattenCriteria(pack: LoadedPack, locale: string) {
  const ar = locale === "ar";
  return pack.standards.flatMap((s) =>
    s.criteria.map((c) => {
      const title = ar && c.titleAr ? c.titleAr : c.titleEn;
      const stdTitle = ar && s.titleAr ? s.titleAr : s.titleEn;
      return {
        criterionId: c.id,
        standardCode: s.code,
        code: c.code,
        title: isPlaceholder(title) ? "" : title,
        // Embed the standard theme + criterion text for a meaningful vector.
        text: [
          isPlaceholder(stdTitle) ? s.code : stdTitle,
          isPlaceholder(title) ? c.code : title,
          isPlaceholder(c.descriptionEn) ? "" : c.descriptionEn,
        ]
          .filter(Boolean)
          .join(". "),
      };
    })
  );
}
