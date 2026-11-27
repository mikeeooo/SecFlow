import type { DecisionLabel, DecisionSource, Outcome } from "../api/types";

// MIA Plăți Instant settles in Moldovan lei only, so every amount is shown in MDL.
const money = new Intl.NumberFormat("en-GB", { style: "currency", currency: "MDL", currencyDisplay: "code", minimumFractionDigits: 2 });
const compactMoney = new Intl.NumberFormat("en-GB", { style: "currency", currency: "MDL", currencyDisplay: "code", notation: "compact", maximumFractionDigits: 1 });
const integer = new Intl.NumberFormat("en-IE");

export const formatMoney = (value: number | null | undefined): string => (value == null ? "—" : money.format(value));
export const formatCompactMoney = (value: number): string => compactMoney.format(value);
export const formatInt = (value: number | null | undefined): string => (value == null ? "—" : integer.format(value));

export function formatPercent(value: number | null | undefined, digits = 1): string {
  return value == null ? "n/a" : `${(value * 100).toFixed(digits)}%`;
}

export function formatScore(value: number | null | undefined): string {
  return value == null ? "—" : value.toFixed(1);
}

/** Simulation and dataset times are shown in UTC so that every view agrees. */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  return `${date.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" })}, ${date.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "UTC" })} UTC`;
}

export function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "UTC" });
}

export function formatDay(iso: string): string {
  return new Date(`${iso.slice(0, 10)}T00:00:00Z`).toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
}

export const SOURCE_TEXT: Record<DecisionSource, string> = {
  confirmed_fraud_evidence: "Confirmed fraud evidence",
  risk_policy: "Anomaly risk policy",
  assessment_safeguard: "Assessment safeguard",
};

export const OUTCOME_TEXT: Record<Outcome, string> = {
  permitted: "Recorded as permitted",
  pending_verification: "Awaiting verification",
  blocked_attempt: "Blocked attempt",
  approved_after_review: "Approved after review",
  rejected_after_review: "Rejected after review",
};

export const LABEL_HINT: Record<DecisionLabel, string> = {
  "Confirmed fraud evidence": "Blocked because trusted security evidence was active at request time.",
  "Preventive risk block": "Blocked by the validated risk policy with corroborating behaviour; not confirmed fraud.",
  "Needs verification": "Held until an analyst approves or rejects it.",
  Permitted: "Below the verification threshold with no active evidence.",
};

export const FAMILY_TEXT: Record<string, string> = {
  velocity: "Velocity",
  amount_deviation: "Amount deviation",
  compromised_access: "Device & sign-in",
  beneficiary_risk: "Beneficiary network",
  funds_flow: "Pass-through funds",
  limit_pressure: "MIA limit pressure",
  identity_change: "Alias rebound",
  payment_instrument: "RTP / QR instrument",
};

/** Human names of the MIA Plăți Instant payment schemes. */
export const SCHEME_TEXT: Record<string, string> = {
  p2p_alias: "P2P by phone alias",
  p2p_iban: "P2P by IBAN",
  p2b_qr_static: "Static merchant QR",
  p2b_qr_dynamic: "Dynamic QR",
  p2b_link: "Payment link",
  request_to_pay: "Request to Pay",
  b2b: "Business to business",
  p2g: "Payment to a public institution",
  b2p_salary: "Salary payment",
  remittance_in: "Incoming remittance",
};

export const formatScheme = (scheme: string | null | undefined): string =>
  scheme ? SCHEME_TEXT[scheme] ?? humanize(scheme) : "—";

export function humanize(value: string): string {
  const text = value.replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function newRequestId(): string {
  const random = typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID().replace(/-/g, "").slice(0, 16)
    : Math.random().toString(36).slice(2, 18);
  return `SIM-${random}`;
}
