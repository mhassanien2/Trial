import assert from "node:assert/strict";
import { test } from "node:test";

import { SOURCE_REGISTRY, findSourceRule } from "../src/lib/calc";
import {
  bmiInputSchema,
  energyInputSchema,
  hydrationLogSchema,
  sleepLogSchema,
} from "../src/lib/validation/schemas";

// ---- Validation (server-side boundary) -----------------------------------------

test("energy input rejects implausible height and weight", () => {
  const valid = {
    sex: "female",
    ageYears: 30,
    weightKg: 65,
    heightCm: 165,
    activityLevel: "light",
    goal: "maintain",
  };
  assert.equal(energyInputSchema.safeParse(valid).success, true);
  assert.equal(energyInputSchema.safeParse({ ...valid, heightCm: 300 }).success, false);
  assert.equal(energyInputSchema.safeParse({ ...valid, weightKg: 5 }).success, false);
});

test("energy input rejects unknown activity levels and goals", () => {
  const base = { sex: "male", ageYears: 30, weightKg: 80, heightCm: 180 };
  assert.equal(energyInputSchema.safeParse({ ...base, activityLevel: "superhuman", goal: "maintain" }).success, false);
  assert.equal(energyInputSchema.safeParse({ ...base, activityLevel: "light", goal: "crash_diet" }).success, false);
});

test("BMI input enforces age and range bounds", () => {
  assert.equal(bmiInputSchema.safeParse({ weightKg: 70, heightCm: 170, ageYears: 40 }).success, true);
  assert.equal(bmiInputSchema.safeParse({ weightKg: 70, heightCm: 170, ageYears: 0 }).success, false);
  assert.equal(bmiInputSchema.safeParse({ weightKg: 70, heightCm: 170, ageYears: 40.5 }).success, false);
});

test("hydration log rejects single entries above 1500 ml and non-integers", () => {
  const at = "2026-10-14T08:00:00.000Z";
  assert.equal(hydrationLogSchema.safeParse({ amountMl: 250, loggedAt: at }).success, true);
  assert.equal(hydrationLogSchema.safeParse({ amountMl: 1600, loggedAt: at }).success, false);
  assert.equal(hydrationLogSchema.safeParse({ amountMl: 250.5, loggedAt: at }).success, false);
});

test("sleep log requires 24-hour clock times and an ISO date", () => {
  const ok = { bedtime: "23:15", wakeTime: "06:45", logDate: "2026-10-14" };
  assert.equal(sleepLogSchema.safeParse(ok).success, true);
  assert.equal(sleepLogSchema.safeParse({ ...ok, bedtime: "11:15pm" }).success, false);
  assert.equal(sleepLogSchema.safeParse({ ...ok, logDate: "14/10/2026" }).success, false);
});

// ---- Source registry integrity ---------------------------------------------------

test("every registry rule has a unique id, citation, year, and uncertainty statement", () => {
  const ids = SOURCE_REGISTRY.map((r) => r.id);
  assert.equal(new Set(ids).size, ids.length, "ids must be unique");
  for (const rule of SOURCE_REGISTRY) {
    assert.ok(rule.citation.length > 10, `${rule.id} needs a citation`);
    assert.ok(rule.year >= 1800 && rule.year <= 2100, `${rule.id} needs a plausible year`);
    assert.ok(rule.uncertainty.length > 10, `${rule.id} must disclose uncertainty`);
  }
});

test("core engine rules are registered", () => {
  for (const id of ["BMI-FORMULA", "RMR-MIFFLIN", "CAL-FLOOR", "AMDR-MACROS", "WATER-AI-2004", "SLEEP-AASM-SRS-2015"]) {
    assert.ok(findSourceRule(id), `missing ${id}`);
  }
});

test("every rule carries a recognised verification status", () => {
  const allowed = new Set(["pending_primary_source_check", "verified_against_primary_source"]);
  for (const rule of SOURCE_REGISTRY) {
    assert.ok(allowed.has(rule.verification), `${rule.id} has unknown status ${rule.verification}`);
  }
});
