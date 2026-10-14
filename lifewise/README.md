# LifeWise AI

A bilingual (Arabic and English) adult wellness platform. This folder is the **calculation core and
data model** of the product. The UI, authentication, and persistence are not built yet.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for scope, phase status, and risks.

> LifeWise AI is a wellness tool. It is not medical advice, diagnosis, or treatment.
> Estimates are not exact measurements.

## What works today

- **Calculation engine** (`src/lib/calc`): BMI and WHO classification, healthy-weight range, waist-to-height
  ratio, Mifflin–St Jeor RMR, TDEE, goal-based calorie targets with safety floors and refusals, macro and
  fibre targets, nutrition aggregation, hydration reference and progress, sleep duration and
  consistency, and trend helpers.
- **Validation** (`src/lib/validation`): Zod schemas for every input the engine accepts.
- **Data model** (`prisma/schema.prisma`): all entities from the brief. Validated and client-generated.
  No migration yet.
- **Source registry** (`src/lib/calc/sources.ts`): each rule with its citation and verification status.

## Requirements

- Node.js 22 or later (developed on 22.22)
- npm

## Run the tests

```bash
cd lifewise
npm install
npm test          # node:test, runs tests/*.test.ts with tsx
npm run typecheck # tsc --noEmit (strict)
```

Schema checks (need a placeholder URL only; no database is contacted):

```bash
DATABASE_URL="postgresql://user:pass@localhost:5432/lifewise_dev" npx prisma validate
DATABASE_URL="postgresql://user:pass@localhost:5432/lifewise_dev" npx prisma generate
```

## Layout

```
docs/ARCHITECTURE.md   architecture, ERD, phase status, risks
prisma/schema.prisma   data model (PostgreSQL)
src/lib/calc/          pure, deterministic calculations (no I/O)
src/lib/validation/    server-side Zod schemas
tests/                 unit, validation, and source-registry tests
```

## Not yet available

- Sign-in, registration, and onboarding
- Any UI or API route
- Database migrations and seed data
- AI features (meal planner, coach)
- Exercise, habit, report, and notification modules

## Configuration

`.env.example` lists the variables the app will need. None are read yet.
