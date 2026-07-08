# Prompt Eval Harness — Project Brief for Claude Code

## Context

Portfolio artifact demonstrating design and ownership of an evaluation + regression-gating harness for an LLM task. This is the third instantiation of an existing design thesis (previously: a YARA detection harness with paired malicious/benign corpora and CI-gated FP thresholds). The owner reviews all output and must be able to defend every line and every design decision live, without preparation. Bias toward simple, boring, reviewable code over clever abstractions. When in doubt, flag the uncertainty instead of guessing.

## Design thesis (do not dilute)

1. Every gate is mechanically checkable — no subjective grading anywhere in the pass/fail path.
2. Every hard case has a deliberate near-miss twin — the corpus is built to probe failure boundaries, not to inflate pass rates.
3. Prompts are versioned artifacts — a prompt change is gated exactly like a code change.

## Task under evaluation

**Default: IOC extraction.** The prompt instructs the model to extract indicators of compromise (IPv4s, domains, URLs, file hashes) from short threat-report-style text passages into a strict JSON schema.

Why this task: outputs are structured and deterministically gradable (no LLM-as-judge needed); near-misses occur naturally (see corpus spec); and it is domain-coherent with the owner's existing detection portfolio.

**Alternative if the owner prefers:** binary classification (suspicious/benign) of short text/code snippets.

> **DECISION CHECKPOINT 0:** Confirm task choice, repo name (proposed: `prompt-eval-harness`), and corpus categories with the owner before scaffolding anything.

## Scope

### In scope (v1)
- Corpus with manifest, output schema, deterministic grader, runner, baseline pinning, regression gates, GitHub Actions CI, README stub.

### Explicit non-goals (scope fences — do not add without owner sign-off)
- No multi-model comparison framework
- No eval UI or dashboard
- No LLM-as-judge
- No prompt auto-optimization / tuning loop
- No RAG, no agent loop
- Single provider (Anthropic API), one primary model string in config

## Repo layout

```
prompt-eval-harness/
├── prompts/
│   ├── v1.md                  # the prompt under evaluation
│   └── v2_regression_demo.md  # deliberately worse (Phase 4 demo)
├── corpus/
│   ├── manifest.yaml          # case id, category, twin_id, input ref, expected ref
│   ├── inputs/                # one text file per case
│   └── expected/              # one JSON file per case (canonical form)
├── harness/
│   ├── schema.py              # pydantic output schema
│   ├── grader.py              # normalization + per-field scoring
│   ├── runner.py              # API calls, caching, baseline pinning
│   └── gates.py               # threshold checks
├── tests/                     # pytest; grader tested on fixtures, no API needed
├── baselines/                 # pinned results per (prompt_version, model)
├── .github/workflows/eval.yml
└── README.md                  # stub only — owner writes the real one
```

## Corpus spec

- 20–25 pairs (40–50 cases total). Small enough to hand-curate; large enough for meaningful per-category metrics.
- Manifest-driven: every case carries `id`, `category`, `twin_id`, input path, expected-output path.
- Near-miss twin categories (each true-positive case gets a paired trap):
  - Defanged IOCs (`hxxp://`, `[.]`) vs. their live forms — decide canonicalization policy
  - Version strings that look like IPv4s (`2.4.41.1`)
  - Git SHAs / UUIDs vs. malware hashes
  - Benign domains in code comments or example text (`example.com`, package registries)
  - Reserved/RFC1918 ranges — extraction policy must be explicit
- Claude Code may draft candidate cases; **the owner approves or edits every case before it enters the corpus.** The corpus judgment is the ownership signal — do not bulk-generate and move on.

> **DECISION CHECKPOINT 1:** Owner reviews the full corpus + canonicalization rules before any API run.

## Grading & metrics

- Normalize model output and expected output to canonical form before comparison (lowercase domains, strip defanging per policy, dedupe).
- Report per-indicator-type precision / recall / F1, plus exact-match rate per case.
- **Failure-category breakdown is required output:** report where failures concentrate (by category and twin type), not just aggregate rates. Distribution matters more than the headline number.

## Gates (CI fails if violated)

1. Schema validity: 100% of outputs parse against the pydantic schema.
2. Recall floor per indicator type — threshold set by owner from baseline data.
3. False-extraction ceiling on the near-miss twin set — threshold set by owner from baseline data.
4. No-regression vs. pinned baseline beyond a stated tolerance.
5. Cost/latency: report-only in v1, not a gate.

> **DECISION CHECKPOINT 2:** Do not invent thresholds. Run the baseline, present the numbers, and let the owner set gates from evidence.

## Nondeterminism policy

- Temperature 0; assume residual drift anyway.
- v1 policy (recommended): single run per case + documented tolerance margin on gates; note the tradeoff vs. N-run majority voting in the README stub as a known limitation.
- Cache responses keyed on `(prompt_hash, model, case_id)`. CI uses cache unless the prompt or corpus changed — a fresh clone must pass `pytest` with no API key using committed fixtures.
- A gate failure must reproduce on re-run before it counts as a regression.

## Tech constraints

- Python 3.11+. Dependencies: `anthropic`, `pytest`, `pydantic`, `pyyaml`. Nothing else — no LangChain or eval frameworks.
- Type hints throughout; small modules; docstrings explain *why*, not *what*.
- API key via env var locally, GitHub secret in CI. Hard per-run cost cap in the runner.

## Phase plan (owner reviews at each checkpoint before the next phase starts)

- **Phase 0** — Confirm task, categories, repo name. STOP for owner.
- **Phase 1** — Schema + grader + corpus manifest, with fixture-based tests. No API calls yet. STOP for owner (Checkpoint 1).
- **Phase 2** — Runner + baseline pinning against `prompts/v1.md`. Present baseline metrics. STOP for owner (Checkpoint 2: thresholds).
- **Phase 3** — Gates + GitHub Actions, triggered on changes to `prompts/` or `corpus/`.
- **Phase 4** — Regression demo: commit `v2_regression_demo.md` on a branch and capture CI going red. README stub + CHANGELOG. Owner writes the final README.

## Definition of done

- Fresh clone passes `pytest` with no API key (cached fixtures).
- A deliberately regressive prompt change turns CI red, reproducibly.
- The owner can state cold: what the harness measures, how every threshold was set, and how she knows it works.
- Final README (owner-written) covers: design thesis, threshold rationale, canonicalization policy, and known limitations (nondeterminism handling, corpus size, single-model scope).

## Working agreement

Stop at every checkpoint. No scope additions without explicit owner sign-off. Prefer boring code. Every file should be reviewable in one sitting. Flag uncertainty; never paper over it.
