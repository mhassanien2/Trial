// Public surface of the calculation engine. UI and API code import from here only.
// This module has no I/O, no database access, and no AI calls. It is deterministic.

export * from "./units";
export * from "./bmi";
export * from "./energy";
export * from "./nutrition";
export * from "./hydration";
export * from "./sleep";
export * from "./trends";
export { SOURCE_REGISTRY, findSourceRule } from "./sources";
export type { SourceRule, VerificationStatus } from "./sources";
