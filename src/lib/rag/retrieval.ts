import { IngestStatus } from "@prisma/client";

import { embedQuery } from "@/lib/ai/embeddings";
import { prisma } from "@/lib/db";

export interface RetrievedChunk {
  chunkId: string;
  documentId: string;
  documentTitle: string;
  page: number | null;
  headingPath: string | null;
  criterionCode: string | null;
  content: string;
  /** fused relevance (reciprocal rank fusion) */
  score: number;
  vectorSimilarity: number | null;
  keywordRank: number | null;
}

export interface HybridSearchResult {
  chunks: RetrievedChunk[];
  /** false ⇒ the app must say it can't answer, never guess */
  confident: boolean;
}

const CANDIDATES = 24;
const RRF_K = 60;

/**
 * Hybrid retrieval: pgvector cosine + tsvector keyword search, fused with
 * reciprocal rank fusion, always scoped to the institution's documents
 * (multi-tenant) with optional documentId / criterionCode metadata filters.
 */
export async function hybridSearch(opts: {
  institutionId: string;
  query: string;
  documentIds?: string[];
  criterionCode?: string;
  limit?: number;
}): Promise<HybridSearchResult> {
  const limit = opts.limit ?? 8;

  // Candidate documents: READY docs of this institution only.
  const docs = await prisma.document.findMany({
    where: {
      institutionId: opts.institutionId,
      ingestStatus: IngestStatus.READY,
      ...(opts.documentIds?.length ? { id: { in: opts.documentIds } } : {}),
    },
    select: { id: true, title: true },
  });
  if (docs.length === 0) return { chunks: [], confident: false };
  const docIds = docs.map((d) => d.id);
  const titleById = new Map(docs.map((d) => [d.id, d.title]));

  const vector = `[${(await embedQuery(opts.query)).join(",")}]`;
  const criterionFilter = opts.criterionCode ?? null;

  type VectorRow = {
    id: string;
    documentId: string;
    content: string;
    page: number | null;
    headingPath: string | null;
    criterionCode: string | null;
    similarity: number;
  };
  const vectorRows = await prisma.$queryRaw<VectorRow[]>`
    SELECT c.id, c."documentId", c.content, c.page,
           c."headingPath", c."criterionCode",
           1 - (c.embedding <=> ${vector}::vector) AS similarity
    FROM "DocumentChunk" c
    WHERE c."documentId" = ANY(${docIds})
      AND c.embedding IS NOT NULL
      AND (${criterionFilter}::text IS NULL OR c."criterionCode" = ${criterionFilter})
    ORDER BY c.embedding <=> ${vector}::vector
    LIMIT ${CANDIDATES}
  `;

  // OR-join significant terms: websearch_to_tsquery ANDs every word
  // (including stopwords like "what"/"ما"), which zeroes out recall for
  // natural-language questions. Ranking still rewards multi-term matches.
  const terms = [
    ...new Set(
      opts.query
        .normalize("NFKC")
        .split(/[^\p{L}\p{N}-]+/u)
        .filter((w) => w.length >= 3)
        .map((w) => w.replace(/'/g, ""))
    ),
  ].slice(0, 16);
  const tsQuery = terms.join(" | ");

  type KeywordRow = {
    id: string;
    documentId: string;
    content: string;
    page: number | null;
    headingPath: string | null;
    criterionCode: string | null;
    rank: number;
    matched: number;
  };
  const keywordRows: KeywordRow[] = tsQuery
    ? await prisma.$queryRaw<KeywordRow[]>`
    SELECT c.id, c."documentId", c.content, c.page,
           c."headingPath", c."criterionCode",
           ts_rank(c.content_tsv, to_tsquery('simple', ${tsQuery})) AS rank,
           (SELECT COUNT(*) FROM unnest(${terms}::text[]) term
            WHERE c.content_tsv @@ to_tsquery('simple', term))::int AS matched
    FROM "DocumentChunk" c
    WHERE c."documentId" = ANY(${docIds})
      AND c.content_tsv @@ to_tsquery('simple', ${tsQuery})
      AND (${criterionFilter}::text IS NULL OR c."criterionCode" = ${criterionFilter})
    ORDER BY matched DESC, rank DESC
    LIMIT ${CANDIDATES}
  `
    : [];

  // Reciprocal rank fusion.
  const fused = new Map<
    string,
    RetrievedChunk & { vRank: number | null; kRank: number | null }
  >();

  vectorRows.forEach((row, i) => {
    fused.set(row.id, {
      chunkId: row.id,
      documentId: row.documentId,
      documentTitle: titleById.get(row.documentId) ?? "Document",
      page: row.page,
      headingPath: row.headingPath,
      criterionCode: row.criterionCode,
      content: row.content,
      score: 1 / (RRF_K + i + 1),
      vectorSimilarity: row.similarity,
      keywordRank: null,
      vRank: i,
      kRank: null,
    });
  });

  keywordRows.forEach((row, i) => {
    const existing = fused.get(row.id);
    if (existing) {
      existing.score += 1 / (RRF_K + i + 1);
      existing.keywordRank = row.rank;
      existing.kRank = i;
    } else {
      fused.set(row.id, {
        chunkId: row.id,
        documentId: row.documentId,
        documentTitle: titleById.get(row.documentId) ?? "Document",
        page: row.page,
        headingPath: row.headingPath,
        criterionCode: row.criterionCode,
        content: row.content,
        score: 1 / (RRF_K + i + 1),
        vectorSimilarity: null,
        keywordRank: row.rank,
        vRank: null,
        kRank: i,
      });
    }
  });

  const chunks: RetrievedChunk[] = [...fused.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((c) => ({
      chunkId: c.chunkId,
      documentId: c.documentId,
      documentTitle: c.documentTitle,
      page: c.page,
      headingPath: c.headingPath,
      criterionCode: c.criterionCode,
      content: c.content,
      score: c.score,
      vectorSimilarity: c.vectorSimilarity,
      keywordRank: c.keywordRank,
    }));

  // Confidence: a keyword hit matching at least two distinct significant
  // query terms, or a single-term query with a hit, or solid vector
  // similarity. (The local dev embedding provider produces weaker
  // similarities, so keyword agreement is the primary signal.)
  const bestMatched = keywordRows[0]?.matched ?? 0;
  const strongKeyword = bestMatched >= Math.min(2, terms.length);
  const topSimilarity = vectorRows[0]?.similarity ?? 0;
  const confident =
    chunks.length > 0 && terms.length > 0 && (strongKeyword || topSimilarity > 0.45);

  return { chunks, confident };
}
