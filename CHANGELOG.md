# Changelog

Every design decision below was ratified by the owner at an explicit checkpoint;
the phase plan and working agreement are in
[prompt_eval_harness_brief.md](prompt_eval_harness_brief.md).

## Phase 0 — scoping (Checkpoint 0)

- Task: IOC extraction (IPv4s, domains, URLs, file hashes → strict JSON).
- Five near-miss twin categories: defanged, version_strings, hash_lookalikes,
  benign_context_domains, reserved_ranges.
- Repo public at github.com/epaneral/prompt-eval-harness.

## Phase 1 — schema, grader, corpus (Checkpoint 1) — commits `52a0be9`..`b9169d7`

- Canonical form is **defanged at rest**: one pinned defang style (`[.]`,
  `hxxp`/`hxxps`, URL paths verbatim); the grader normalizes representation
  only — extraction judgment lives in the prompt and expected files.
- Judgment policies: non-global IPv4s excluded (`ipaddress.is_global`); URL
  hosts also extracted into domains/ipv4; flat `hashes` list; RFC 2606/6761
  names and incidental benign-context domains excluded.
- Corpus: 22 owner-approved twin pairs (44 cases), every trap a single edit
  from its positive twin; integrity suite enforces manifest consistency,
  canonical form, judgment policies, and non-empty expected files.

## Phase 2 — runner and baselines (Checkpoint 2) — commits `71c5ed2`..`a273bee`

- Committed response cache keyed on (prompt_sha256, model, case_id) with
  input-hash staleness validation; fresh clones grade with no API key.
- Hard $0.50 per-run cost cap. Primary model claude-sonnet-4-6 (temperature 0);
  claude-haiku-4-5 and claude-opus-4-8 selectable, pricing-gated.
- Fence policy is per-model and baseline-derived: Sonnet/Opus strict (0/44
  fenced at baseline), Haiku normalize (44/44); fenced responses counted for
  every model regardless of policy.
- Schema-invalid responses grade as empty extractions; denominators labeled;
  trap metric conditioned on schema-valid traps, fail-closed when undefined.
- Baselines: Sonnet and Opus 4.8 saturate the corpus (44/44 exact match, 0%
  trap leak). Haiku: 93.2% exact match, 9.1% trap leak, failures concentrated
  in benign_context_domains — the corpus discriminates on judgment.
- Drift study (baselines/drift_v1_claude-sonnet-4-6.json): 5 independent
  uncached runs, 220 case-samples, zero deviations, identical token usage.

## Phase 3 — gates and CI — commit `855a22e`

- Gates (all evidence-derived, all fail-closed on undefined metrics):
  schema validity 100% under the model's recorded fence policy; per-type
  recall floor 1.0; trap ceiling 0% of schema-valid traps; no metric below
  the pinned baseline.
- Ratified reproduce-on-fresh-samples rule automated as
  `python -m harness.gates --refresh-failing`: a red must repeat after the
  failing cases' cache entries are deleted and resampled.
- CI: `tests` job on every push/PR with no API key; `gates` job only when the
  eval surface changes.

## Phase 4 — regression demo

- `prompts/v2_regression_demo.md`: a plausible bad edit — the judgment rules
  deleted and a recall-maximizing instruction added ("when in doubt, include
  it"). Applied to the gated prompt on the `regression-demo` branch
  (do not merge); the pull request exists to stay red.
- Measured damage (single run, claude-sonnet-4-6, $0.10): schema validity
  still 100% — the regression is pure judgment — while **12 of 22 traps leak
  (54.5%)**: all benign_context_domains, all reserved_ranges, 3 of 4
  hash_lookalikes. Recall stays 1.000 everywhere; precision falls on every
  type (ipv4 .886, domains .872, urls .818, hashes .889); exact match 72.7%.
  A recall-only eval would score this edit as an improvement.
- Notable: version_strings and defanged traps still held — model judgment
  resists IPv4-shaped version strings even when told to over-include.
