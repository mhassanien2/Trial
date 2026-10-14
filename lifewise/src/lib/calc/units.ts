// Unit conversions. Every calculation in the engine works in SI units
// (kg, cm, years, ml, minutes). Imperial input is converted at the boundary.

export const KG_PER_LB = 0.45359237;
export const CM_PER_IN = 2.54;

export function lbToKg(pounds: number): number {
  return pounds * KG_PER_LB;
}

export function kgToLb(kilograms: number): number {
  return kilograms / KG_PER_LB;
}

export function ftInToCm(feet: number, inches: number): number {
  return (feet * 12 + inches) * CM_PER_IN;
}

export function cmToFtIn(centimeters: number): { feet: number; inches: number } {
  const totalInches = centimeters / CM_PER_IN;
  const feet = Math.floor(totalInches / 12);
  const inches = Math.round((totalInches - feet * 12) * 10) / 10;
  return { feet, inches };
}
