# AI Prompts

Every prompt used by AccreditGenius lives here as a versioned, editable
TypeScript module — prompts are NEVER inlined in components or routes.

Conventions:
- Each module exports `PROMPT_VERSION` (e.g. `standards-qa@v1`). The
  version string is written to the `AiInteraction` audit table with every
  call, so any historical output can be traced to the exact prompt text.
- Bump the version suffix whenever the wording changes materially.
- Prompts must enforce the product's non-negotiables: cited answers only,
  no fabricated standards content, explicit refusal when retrieval
  confidence is low.
