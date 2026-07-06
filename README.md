# AccreditGenius

A web-based **Quality & Accreditation Assistant** for higher-education
institutions. It supports the full quality cycle (Plan → Do → Check → Act)
and helps QA units, program coordinators, and faculty produce, review, and
manage accreditation documentation against national and international
standards.

Bilingual (English + Arabic, full RTL). Multi-tenant from day one.

## Capabilities

1. **Standards Knowledge Base** — pre-loaded packs for Saudi Arabia (NCAAA
   program & institutional, ETEC, SAQF) and Egypt (NAQAAE, Egyptian NQF),
   stored as structured JSON in `data/standards/`. Any other country: users
   upload their own standards/NQF documents as a selectable custom pack.
   Official wording is loaded by replacing `TODO:OFFICIAL_TEXT` placeholders.
2. **RAG Document Intelligence** — PDF/DOCX ingestion → heading-aware
   chunking → embeddings → pgvector; hybrid retrieval (vector + tsvector
   keyword + metadata). Every AI answer cites source chunks; low-confidence
   retrieval is refused rather than guessed.
3. **Template-Locked Document Generator** — upload an official DOCX template;
   a parser turns it into a JSON schema that drives both a guided wizard and
   a fidelity-preserving DOCX/PDF export. Unfilled fields export as
   `[REQUIRES INPUT]` — never fabricated.
4. **AI Document Reviewer** — rubric-based, criterion-by-criterion compliance
   analysis with a 0–100 readiness score, RAG-cited findings
   (Met / Partially Met / Not Met), recommendations, and a DOCX report.
5. **Quality Cycle Workspace** — per-program PDCA board, evidence repository
   with a criterion coverage heat map, and an improvement-action tracker.
6. **Differentiators** — Mock Accreditation Panel (3 reviewer personas +
   predicted site-visit questions), Cross-Standard Mapper (overlap/gap matrix
   across two packs), and a per-program readiness trendline.

## Tech stack

- **Next.js 16** (App Router, TypeScript), Tailwind CSS v4, shadcn/ui
- **PostgreSQL + pgvector**, Prisma ORM
- **NextAuth v5** (email/password + optional Google), RBAC: Admin, QA
  Director, Program Coordinator, Faculty, Reviewer (read-only)
- **Anthropic API** (`claude-sonnet-4-6`) for generation/review; embeddings
  behind a swappable `lib/ai/embeddings.ts` (Voyage / OpenAI / local dev)
- **pdf-parse / mammoth** for extraction, **docx** for DOCX generation
- **StorageProvider** interface (local in dev, S3/R2 later)
- **DB-backed job queue** for ingestion/review (no Redis in v1)

Every AI generation/review is written to an audit trail (`AiInteraction`,
`AuditLog`): who, when, model, prompt version, input hash, output.

## Local setup

### Prerequisites
- Node 20+ and pnpm 10+
- PostgreSQL 16 with the `vector` and `pg_trgm` extensions

```bash
# 1. Postgres + extensions
createdb accreditgenius
psql -d accreditgenius -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;"

# 2. Install deps
pnpm install

# 3. Configure environment
cp .env.example .env
#   set DATABASE_URL, AUTH_SECRET (openssl rand -base64 32)
#   optional: ANTHROPIC_API_KEY, VOYAGE_API_KEY/OPENAI_API_KEY,
#             AUTH_GOOGLE_ID / AUTH_GOOGLE_SECRET

# 4. Migrate + seed demo data
pnpm db:deploy      # or: pnpm db:migrate
pnpm db:seed

# 5. Run
pnpm dev            # http://localhost:3000
```

### Demo accounts
After `pnpm db:seed` (password `Demo1234!` for all):

| Email | Role |
| --- | --- |
| admin@demo.edu | Admin |
| qa@demo.edu | QA Director |
| coordinator@demo.edu | Program Coordinator |
| faculty@demo.edu | Faculty |
| reviewer@demo.edu | Reviewer (read-only) |

Seed also creates: Demo University, PharmD + MBBCH programs, the 7 standards
packs, a TP-153 Course Specification template, an ingested PharmD SSR, a PDCA
board, evidence links, improvement actions, and a readiness trendline.

### AI keys are optional
Without `ANTHROPIC_API_KEY` the app degrades gracefully: Q&A returns cited
passages (no synthesis), the reviewer uses an evidence-coverage heuristic,
and the mock panel shows an "AI required" placeholder — nothing is
fabricated. Without an embeddings key a deterministic local embedder is used
(dev quality only).

## Testing

```bash
pnpm test        # unit + integration (node:test): parser fidelity, rate limiter
pnpm test:e2e    # Playwright: the 3 critical flows (needs a seeded, running app)
```

The e2e suite covers **ingest → ask**, **template → export**, and
**upload → review**. Start a seeded app (`pnpm build && pnpm start` after
`pnpm db:seed`) then run `pnpm test:e2e` (reuses the running server).

## Background jobs
Ingestion, template parsing, reviews, and mock-panel runs are queued in the
`Job` table and executed by the in-process runner. Uploads kick it
automatically; you can also drain it via `POST /api/jobs/run`
(authenticated, or with a `JOBS_RUN_SECRET` bearer token for an external
scheduler / cron).

## Deployment

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the full guide. In short,
because the app needs Postgres + pgvector, an object store, and a Node
runtime for file processing, deploy it as:

- **App** → Cloudflare (Workers/Pages via OpenNext) or any Node host
- **Database** → managed Postgres with pgvector (e.g. Neon)
- **Storage** → Cloudflare R2 (S3-compatible `StorageProvider`)
- **Jobs** → Cloudflare Cron Trigger hitting `POST /api/jobs/run`

## Project layout

```
data/standards/        seeded standards packs (JSON, TODO:OFFICIAL_TEXT)
prisma/                schema, migrations, seed
src/app/[locale]/      localized App Router pages (RTL-aware)
src/app/api/           route handlers (tenant-scoped, zod-validated)
src/components/         UI (shadcn/ui + feature components)
src/lib/ai/            client, embeddings, versioned prompts
src/lib/ingest/        extraction + heading-aware chunking
src/lib/rag/           hybrid retrieval
src/lib/templates/     DOCX parser + exporter (the product's core)
src/lib/review/        rubric engine + report
src/lib/panel/         mock accreditation panel
src/lib/mapper/        cross-standard mapper
src/lib/jobs/          DB-backed queue + handlers
tests/                 unit + Playwright e2e
docs/                  ERD, deployment guide
```

## Non-negotiables (enforced)
- No fabricated standards content; AI outputs about standards carry citations.
- Template fidelity preserved on export; missing content → `[REQUIRES INPUT]`.
- Multi-tenant: every query scoped by `institutionId`.
- Prompts are versioned files under `src/lib/ai/prompts/`.
- Arabic/RTL correctness across screens and DOCX exports.
- Full AI audit trail for accreditation traceability.
