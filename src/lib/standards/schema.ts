import { z } from "zod";

/**
 * JSON schema for standards packs shipped in /data/standards.
 *
 * IMPORTANT: content marked "TODO:OFFICIAL_TEXT" is a structural
 * placeholder. The application NEVER treats placeholder text as real
 * standards content — the RAG layer excludes it from citations and the
 * UI renders it as "official text pending".
 */

export const OFFICIAL_TEXT_PLACEHOLDER = "TODO:OFFICIAL_TEXT";

export const indicatorSchema = z.object({
  code: z.string().min(1),
  textEn: z.string().min(1),
  textAr: z.string().optional(),
});

export const evidenceRequirementSchema = z.object({
  textEn: z.string().min(1),
  textAr: z.string().optional(),
});

export const criterionSchema = z.object({
  code: z.string().min(1),
  titleEn: z.string().min(1),
  titleAr: z.string().optional(),
  descriptionEn: z.string().optional(),
  descriptionAr: z.string().optional(),
  indicators: z.array(indicatorSchema).default([]),
  evidenceRequirements: z.array(evidenceRequirementSchema).default([]),
});

export const standardSchema = z.object({
  code: z.string().min(1),
  titleEn: z.string().min(1),
  titleAr: z.string().optional(),
  descriptionEn: z.string().optional(),
  descriptionAr: z.string().optional(),
  criteria: z.array(criterionSchema).default([]),
});

export const standardsPackFileSchema = z.object({
  code: z.string().min(1), // unique with version, e.g. "NCAAA-PROG-2022"
  country: z.string().length(2), // ISO alpha-2
  nameEn: z.string().min(1),
  nameAr: z.string().optional(),
  version: z.string().default("1.0"),
  description: z.string().optional(),
  standards: z.array(standardSchema),
});

export type StandardsPackFile = z.infer<typeof standardsPackFileSchema>;

export function isPlaceholder(text: string | null | undefined): boolean {
  return !text || text.includes(OFFICIAL_TEXT_PLACEHOLDER);
}
