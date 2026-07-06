# Deployment Guide

AccreditGenius is a full Next.js 16 app that needs three stateful pieces:

1. **PostgreSQL with `pgvector`** — RAG embeddings + relational data
2. **Object storage** — uploaded/generated files (behind `StorageProvider`)
3. **A Node runtime** — PDF/DOCX processing (`pdf-parse`, `mammoth`, `docx`)
   and the background job runner

This guide covers deploying to **Cloudflare** (the target) plus a simpler
Node-host fallback. GitHub hosts the source; Cloudflare builds and serves it.

---

## Option A — Cloudflare (Workers via OpenNext)

Cloudflare Workers don't run Postgres or a persistent filesystem, so pair the
Worker with managed Postgres and R2.

### 1. Provision managed Postgres + pgvector

Use a provider that supports `pgvector` over TCP, e.g. **Neon**:

```sql
-- once, in your Neon database
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

Copy the **pooled** connection string (Neon → Connection Details → Pooled).
Prisma runs over TCP from the Worker via Neon's pooler.

Apply the schema from your machine or CI:

```bash
DATABASE_URL="postgresql://…neon…/accreditgenius?sslmode=require" \
  pnpm db:deploy
DATABASE_URL="…" pnpm db:seed     # optional demo data
```

### 2. Create an R2 bucket for storage

```bash
npx wrangler r2 bucket create accreditgenius-uploads
```

R2 is S3-compatible. Set `STORAGE_DRIVER=s3` and the R2 credentials (below).
A ready `S3StorageProvider` can be added behind the existing
`StorageProvider` interface in `src/lib/storage/` — the interface already
abstracts `put/get/delete/exists`, so only that one file changes.

### 3. OpenNext + Wrangler tooling (already wired)

`@opennextjs/cloudflare` and `wrangler` are dev dependencies, and
`wrangler.jsonc` + `open-next.config.ts` are in the repo root. The
Cloudflare Worker bundle builds successfully (`pnpm cf:build` →
`.open-next/worker.js`).

> **Edge-compatible by design.** Next 16's `proxy` (middleware) only runs
> on the Node.js runtime, which Cloudflare Workers reject. This app
> therefore ships **no middleware**: locale routing is handled by the
> `[locale]` segment + a root redirect (`src/app/page.tsx`), and auth is
> enforced authoritatively server-side in every page/route via
> `requireTenant` / `requireTenantWith`. Nothing about security depends on
> an edge pre-gate.

### 4. Configure secrets

In the Cloudflare dashboard (Workers → your worker → Settings → Variables)
or via `wrangler secret put`:

| Secret | Notes |
| --- | --- |
| `DATABASE_URL` | Neon pooled connection string |
| `AUTH_SECRET` | `openssl rand -base64 32` |
| `AUTH_URL` | your production URL, e.g. `https://accreditgenius.example.workers.dev` |
| `ANTHROPIC_API_KEY` | optional (AI features) |
| `VOYAGE_API_KEY` or `OPENAI_API_KEY` | optional (embeddings) |
| `AUTH_GOOGLE_ID` / `AUTH_GOOGLE_SECRET` | optional (Google sign-in) |
| `STORAGE_DRIVER` | `s3` |
| `S3_BUCKET`, `S3_ENDPOINT`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` | R2 values |
| `JOBS_RUN_SECRET` | bearer token for the cron job runner |

### 5. Build & deploy

```bash
pnpm cf:build      # opennextjs-cloudflare build  → .open-next/worker.js
pnpm cf:deploy     # build + wrangler deploy       (needs CLOUDFLARE_API_TOKEN)
# local preview on workerd:
pnpm cf:preview    # build + wrangler dev
```

Or connect the GitHub repo in the Cloudflare dashboard (Workers & Pages →
Create → connect to Git) with build command `pnpm cf:build` and deploy
command `pnpm exec wrangler deploy` — every push to the deployment branch
ships.

### 6. Run background jobs on a schedule

Workers are request-scoped, so drain the queue with a **Cron Trigger**.
`wrangler.jsonc` includes a `triggers.crons` entry; point it at a small
scheduled handler (or an external uptime cron) that calls:

```
POST https://<your-worker>/api/jobs/run
Authorization: Bearer <JOBS_RUN_SECRET>
```

Uploads also kick the runner in-process, so the cron is a safety net for
retries and any work queued without an immediate request.

> **Note on the Node runtime.** `pdf-parse`/`pdfjs` and `docx` need Node
> APIs. OpenNext runs Next.js server code on Workers with `nodejs_compat`
> (enabled in `wrangler.jsonc`). If a specific library misbehaves on
> Workers, run ingestion/export in a small companion **Node** service (or
> Cloudflare Container) and keep the web app on Workers — the job queue
> already decouples this cleanly.

---

## Option B — Node host (simplest, fully compatible)

Any Node host (Fly.io, Render, Railway, a VM, or Cloudflare Containers)
runs the app as-is:

```bash
pnpm install --prod=false
pnpm db:deploy
pnpm build
pnpm start        # serves on $PORT (default 3000)
```

Provide the same env vars as above; `STORAGE_DRIVER=local` works if the host
has a persistent disk mounted at `UPLOADS_DIR`, otherwise use `s3`/R2.
Schedule `POST /api/jobs/run` (with `JOBS_RUN_SECRET`) every minute via the
host's cron.

---

## CI

`.github/workflows/ci.yml` runs typecheck, lint, and unit/integration tests
on every push (Postgres + pgvector service container). Wire deployment by
adding a deploy job gated on `main` with your Cloudflare API token in
`CLOUDFLARE_API_TOKEN`.

## Production checklist
- [ ] `AUTH_SECRET` set to a strong random value; `AUTH_URL` = prod URL
- [ ] Postgres has `vector` + `pg_trgm`; migrations applied (`pnpm db:deploy`)
- [ ] `STORAGE_DRIVER=s3` with R2 credentials (no ephemeral local disk)
- [ ] `JOBS_RUN_SECRET` set and a scheduled trigger drains `/api/jobs/run`
- [ ] Rate-limit windows reviewed (`src/lib/rate-limit.ts`)
- [ ] Google OAuth redirect URIs include the prod domain (if enabled)
