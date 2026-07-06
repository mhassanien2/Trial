# AccreditGenius — Entity Relationship Diagram

Generated from `prisma/schema.prisma` (Phase 1). Every domain table is
scoped by `institutionId` (multi-tenant). Official standards packs are the
one exception: `StandardsPack.institutionId` is nullable — `null` marks a
globally shipped pack (NCAAA, NAQAAE, …).

```mermaid
erDiagram
    Institution ||--o{ User : "has"
    Institution ||--o{ Program : "has"
    Institution ||--o{ Document : "owns"
    Institution ||--o{ Template : "owns"
    Institution ||--o{ AuditLog : "records"
    Institution ||--o{ AiInteraction : "records"
    Institution ||--o{ Job : "queues"
    Institution |o--o{ StandardsPack : "custom packs (null = official)"

    User ||--o{ Account : "oauth"
    User ||--o{ Session : "sessions"
    User ||--o{ ProgramMember : "memberships"
    Program ||--o{ ProgramMember : "members"

    StandardsPack ||--o{ Standard : "contains"
    Standard ||--o{ Criterion : "contains"
    Criterion ||--o{ Indicator : "has"
    Criterion ||--o{ EvidenceRequirement : "requires"

    Document ||--o{ DocumentChunk : "chunked into (pgvector + tsvector)"
    Document |o--|| StandardsPack : "source of custom pack"
    Document |o--|| Template : "source of template"
    User |o--o{ Document : "uploaded by"
    Program |o--o{ Document : "belongs to"

    Template ||--o{ GeneratedDocument : "instantiates"
    Program ||--o{ GeneratedDocument : "produces"

    Document ||--o{ Review : "review subject"
    StandardsPack ||--o{ Review : "against"
    Review ||--o{ ReviewFinding : "yields"
    Criterion ||--o{ ReviewFinding : "assessed"

    Program ||--o{ EvidenceLink : "evidence"
    Document ||--o{ EvidenceLink : "file"
    Criterion ||--o{ EvidenceLink : "tagged to"

    Program ||--o{ PlanItem : "PDCA board"
    GeneratedDocument |o--o{ PlanItem : "tracks"
    Program ||--o{ ImprovementAction : "actions"
    ReviewFinding |o--o{ ImprovementAction : "spawned by"
    User |o--o{ ImprovementAction : "owner"

    Program ||--o{ ReadinessSnapshot : "trendline"
    StandardsPack ||--o{ ReadinessSnapshot : "scored against"

    Criterion ||--o{ CriterionMapping : "maps from"
    Criterion ||--o{ CriterionMapping : "maps to"
    Program ||--o{ MockPanelRun : "simulations"
    Document ||--o{ MockPanelRun : "SSR input"
```

## Key columns per table (abridged)

| Table | Tenant scope | Notable columns |
| --- | --- | --- |
| `User` | `institutionId` | `role` (ADMIN / QA_DIRECTOR / PROGRAM_COORDINATOR / FACULTY / REVIEWER), `passwordHash`, `locale` |
| `Program` | `institutionId` | `code`, `nameEn/nameAr`, `degreeLevel`, `nqfLevel` |
| `StandardsPack` | nullable (null = official) | `origin` OFFICIAL/CUSTOM, `country`, `code`, `version`, `sourceDocumentId` |
| `Standard` → `Criterion` → `Indicator` | via pack | bilingual `titleEn/titleAr`, NCAAA-style `code` |
| `Document` | `institutionId` | `kind`, `storageKey`, `sha256`, `ingestStatus`, `metadata` JSON |
| `DocumentChunk` | via document | `embedding vector(1536)` (HNSW), `content_tsv` generated tsvector (GIN), `page`, `headingPath`, `criterionCode` |
| `Template` | `institutionId` | `formCode` (e.g. TP-153), `schemaJson` (drives wizard + export) |
| `GeneratedDocument` | `institutionId` | `contentJson`, `status`, `exportDocxKey/exportPdfKey` |
| `Review` / `ReviewFinding` | `institutionId` | `readinessScore` 0–100, `verdict` MET/PARTIALLY_MET/NOT_MET, `citations` JSON |
| `EvidenceLink` | `institutionId` | unique (document, criterion, program) — powers the coverage heat map |
| `AuditLog` / `AiInteraction` | `institutionId` | who/when/model/`promptVersion`/`inputHash` — accreditation traceability |
| `Job` | `institutionId?` | DB-backed queue: `status`, `attempts`, `runAfter`, `lockedAt/lockedBy` |
