# Standards Packs (seed data)

Each JSON file in `sa/` (Saudi Arabia) and `eg/` (Egypt) defines one
official standards pack that ships with AccreditGenius. Files are
validated against `src/lib/standards/schema.ts` and loaded by
`pnpm db:seed`.

## Structure

```json
{
  "code": "NCAAA-PROG-2022",
  "country": "SA",
  "nameEn": "…",
  "nameAr": "…",
  "version": "2022",
  "standards": [
    {
      "code": "S1",
      "titleEn": "…",
      "titleAr": "…",
      "criteria": [
        {
          "code": "1-1",
          "titleEn": "…",
          "descriptionEn": "…",
          "indicators": [{ "code": "…", "textEn": "…" }],
          "evidenceRequirements": [{ "textEn": "…" }]
        }
      ]
    }
  ]
}
```

## TODO:OFFICIAL_TEXT placeholders

Every field whose official wording has not yet been supplied is marked
`TODO:OFFICIAL_TEXT`. **The app never presents placeholder text as real
standards content**: it is excluded from RAG citations and rendered as
"official text pending" in the UI.

To load the official text: replace the placeholder strings with the
licensed official wording (keeping codes and structure), bump the pack
`version` if the source document version changed, and re-run
`pnpm db:seed`.

Structural notes:
- Standard/criterion codes follow each framework's own numbering.
- The NCAAA document-code taxonomy (DI-xxx/TI-xxx institutional,
  DP-xxx/TP-xxx program) from the official M108 catalog is used for
  template `formCode` values elsewhere in the app.
