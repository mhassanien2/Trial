-- Hybrid search support for DocumentChunk:
-- 1. Generated tsvector column for keyword search ('simple' config so Arabic
--    and English tokens are both indexed without stemming surprises).
-- 2. GIN index on the tsvector.
-- 3. HNSW index for cosine similarity over pgvector embeddings.

ALTER TABLE "DocumentChunk"
  ADD COLUMN "content_tsv" tsvector
  GENERATED ALWAYS AS (to_tsvector('simple', coalesce("content", ''))) STORED;

CREATE INDEX "DocumentChunk_content_tsv_idx"
  ON "DocumentChunk" USING GIN ("content_tsv");

CREATE INDEX "DocumentChunk_embedding_idx"
  ON "DocumentChunk" USING hnsw ("embedding" vector_cosine_ops);
