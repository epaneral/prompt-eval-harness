# prompt-eval-harness

An evaluation and regression-gating harness for an LLM extraction task — built to answer the question every team shipping LLM-powered automation eventually hits: how do you know a prompt change made things better, and how do you stop one that made things worse?

The harness owns the full evaluation stack: a hand-curated adversarial corpus, deterministic grading with no LLM-as-judge, gate thresholds derived from measured baselines and a drift study, and CI gates that turn red — reproducibly — when the prompt regresses.

The concrete task is extracting indicators of compromise (IPs, domains, URLs, file hashes) from security telemetry like proxy logs and incident reports. The design generalizes to any structured-extraction task where near-misses matter more than obvious cases: every trap case in the corpus differs from a true positive by exactly one edit (a git SHA that isn't a file hash, a private IP, a documentation domain), so the benchmark measures model judgment rather than pattern matching.

The proof it works is a deliberately broken PR. The regression-demo branch commits a plausible "improvement" — a recall-maximizing prompt edit that holds recall at 1.000 and schema validity at 100% while leaking 54.5% of trap cases. A recall-only eval scores this as a gain. The CI gate caught it, automatically re-sampled the failing cases fresh, and reproduced identical metrics. That pull request stays open on purpose: → [the red PR](https://github.com/epaneral/prompt-eval-harness/pull/1).

All corpus indicators are fabricated; any resemblance to registered infrastructure is coincidental. This corpus is not threat intelligence.

---

Status: Phases 1–4 complete — gates live in CI, regression demo on the `regression-demo` branch. Decision history in [CHANGELOG.md](CHANGELOG.md); design brief in [prompt_eval_harness_brief.md](prompt_eval_harness_brief.md).
