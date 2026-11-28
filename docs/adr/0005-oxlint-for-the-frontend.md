# 0005. Lint the frontend with oxlint, not ESLint

Status: accepted, 2026-11-28.

## Context

The audit proposed an ESLint flat config with typescript-eslint and the react-hooks plugin. The frontend uses TypeScript 7.0.2. typescript-eslint parses TypeScript through the TypeScript JavaScript API, and TypeScript 7 does not ship that API: `ts.createSourceFile` is undefined. typescript-eslint 8.74 also declares `typescript >=4.8.4 <6.1.0`. `npm install` fails with ERESOLVE, and npm `overrides` does not fix that.

oxlint parses TypeScript and JSX itself. It has the ESLint correctness rules, the TypeScript rules and both classic react-hooks rules, `rules-of-hooks` and `exhaustive-deps`.

## Decision

Lint `frontend/` with oxlint. `frontend/.oxlintrc.json` turns on the correctness category for the `eslint`, `typescript` and `react` plugins, plus the two react-hooks rules. `npm run lint` runs `oxlint --deny-warnings`, and CI runs it as a gate.

The React Compiler rules in oxlint's React correctness category are turned off: `set-state-in-effect`, `immutability` and `refs`. They report 5 errors today. The audit did not cover them, and fixing them changes component code.

## Consequences

- ESLint plugins that oxlint does not implement are not available.
- If typescript-eslint gains support for TypeScript 7, moving to ESLint is a config change plus a lockfile update.
- The 5 React Compiler findings are a possible follow-up.
