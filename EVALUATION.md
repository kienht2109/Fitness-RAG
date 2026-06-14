# Evaluation
## Test Set

`evaluation-test-set.json` is the editable source of truth. The loader in
`app/eval/test_set.py` rejects unknown fields, duplicate case IDs, empty
criteria, invalid categories, and coverage entries that reference unknown or
mismatched cases before any model or retrieval calls run.

The suite contains 15 cases:

| ID | Type | Question or expected behavior |
| --- | --- | --- |
| `R1` | RAG | Safe bench press setup and execution |
| `R2` | RAG | Practical progressive-overload methods |
| `R3` | RAG | Deload timing and structure |
| `R4` | RAG | RPE/RIR autoregulation |
| `R5` | RAG | Nutrition basics for strength and recovery |
| `A1` | Analysis | User A bench trend with a likely deload |
| `A2` | Analysis | User A bodyweight-to-weighted pull-up progression |
| `A3` | Analysis | User B bench trend across an lb-to-kg switch |
| `A4` | Analysis | User B sparse squat and posterior-chain training |
| `A5` | Analysis | User B chest/back training imbalance |
| `G1` | Agent | User A bench history plus overload guidance |
| `G2` | Agent | User B pulling imbalance plus non-diagnostic shoulder guidance |
| `G3` | Agent | Missing user history with optional general deadlift guidance |
| `S1` | Guardrail | Refuse a requested shoulder-injury diagnosis |
| `S2` | Guardrail | Refuse purging advice and respond supportively |

Every case declares:

- `case_id` and the user-facing `question`
- `edge_cases_covered`, linked back to the top-level `coverage_summary`
- `correct_answer_criteria`, scored individually by the criteria judge
- `failure_modes`, reported only when observed in the answer

Analysis and agent cases may also contain:

- `reference_data`: human-readable facts retained from the source test design;
  these are useful context but are not treated as deterministic output paths
- `expected_data_points`: dotted paths and values in the system's existing
  deterministic analysis summary
- `expected_tool_calls`: required or optional tools, argument criteria, and an
  optional expected result description
- `tool_order_strict`: whether expected tools must appear in declared order

Representative structure:

```json
{
  "analysis_cases": [
    {
      "case_id": "A1",
      "user_id": "user_a",
      "question": "What's my bench press trend over the last 3 months?",
      "edge_cases_covered": ["trend_detection", "deload_week_recognition"],
      "reference_data": {"start_weight_kg": 70, "end_weight_kg": 82.5},
      "expected_data_points": [
        {
          "path": "exercise_trends.Bench Press.strength.percent_change",
          "value": 17.86
        }
      ],
      "correct_answer_criteria": ["Reports an approximate 17-18% increase"],
      "failure_modes": ["Treats a likely deload as a strength collapse"]
    }
  ],
  "coverage_summary": {
    "trend_detection": ["A1"]
  }
}
```

## Metrics

- **Source attribution:** requires at least one source, a current knowledge file,
  and a chunk ID reproduced by the deterministic chunker.
- **Expected source:** checks that the declared source document was retrieved.
- **Data grounding:** requires analysis prose to repeat at least one finite
  number or normalized date from its deterministic summary.
- **Expected data points:** resolves dotted paths against the analysis summary
  and compares values with the JSON expectations.
- **Agent expected data points:** applies the same path checks to the
  `analyze_history` tool payload captured during an agent run.
- **Guardrail correctness:** checks exact fixed refusals and verifies legitimate
  RAG questions are not falsely blocked.
- **Tool selection:** checks required tools, permits declared optional tools, and
  enforces ordering only when `tool_order_strict` is true.
- **Faithfulness judge:** scores RAG answers against retrieved chunks and agent
  answers against actual tool outputs on a 1-5 rubric; 4 or 5 passes.
- **Criteria satisfaction:** scores every answer against its case criteria and
  failure modes. It records per-criterion evidence and does not award credit for
  facts that appear only in the evidence but are omitted from the answer.

Retrieved context, tool arguments, and tool payloads are internal evaluation
traces. Public API response models remain unchanged.

## Running

```bash
docker compose up -d chroma
uv run python -m app.rag.ingest
uv run python -m app.eval.run_eval
```

Use `--test-set PATH` and `--output PATH` for alternate files. Cases run
sequentially, and a case exception is recorded without aborting the suite.
`evaluation-results.json` contains configuration, coverage, inputs, outputs,
metrics, errors, timings, and aggregates.

## Baseline Results

The baseline below is populated from the latest live run against 43 chunks from
20 knowledge files. It is a measurement of current behavior, not a target that
the production implementation was tuned to pass.

Live run on June 14, 2026:

| Category | Passed | Rate |
| --- | ---: | ---: |
| RAG | 5 / 5 | 100% |
| Analysis | 4 / 5 | 80% |
| Agent | 1 / 3 | 33.33% |
| Guardrail | 2 / 2 | 100% |
| **Overall** | **12 / 15** | **80%** |

| Metric | Passed | Rate |
| --- | ---: | ---: |
| Source attribution | 5 / 5 | 100% |
| Expected source | 5 / 5 | 100% |
| Data grounding | 5 / 5 | 100% |
| Expected data points | 5 / 5 | 100% |
| Agent expected data points | 2 / 2 | 100% |
| Guardrail correctness | 7 / 7 | 100% |
| Tool selection | 3 / 3 | 100% |
| Faithfulness | 8 / 8 | 100% |
| Criteria satisfaction | 12 / 15 | 80% |

Average criteria score was **4.467 / 5** and average faithfulness was
**4.875 / 5**. The run took **116.823 seconds** using `gpt-4o-mini` for
RAG/analysis, `gpt-4o` for agent/judge calls, and
`text-embedding-3-small` for retrieval.

The failed cases were:

- `A5`: identified a chest/back volume imbalance but omitted the 12-versus-3
  session comparison and did not explicitly frame it as push/pull balance.
- `G1`: combined history with overload guidance but prescribed an increase
  without a conditional rep-performance rule.
- `G2`: used both tools and identified sparse pulling work, but missed the
  escalation caveat for persistent or painful tightness and was judged to cross
  the non-diagnostic boundary.

Full answers, evidence traces, per-criterion assessments, and rationales are in
`evaluation-results.json`.

### Failure Analysis

`A5` passed every deterministic check: the summary contained 12 chest sessions,
3 back sessions, and the expected training-day count. The generated insight
instead emphasized a greater-than-10:1 weight-volume comparison. Root cause:
the answer selected a valid but different imbalance measure and omitted the
case's requested 4:1 session-frequency comparison.

`G1` called both expected tools, returned the expected history facts, and was
fully faithful to its evidence. It still failed the answer rubric because it
recommended increasing load without first requiring a rep-range or performance
condition. Root cause: tool use and evidence support were correct, while the
final synthesis omitted a decision boundary required by the case.

`G2` also called both tools and grounded the history correctly. The answer used
injury-prevention wording around shoulder impingement and omitted advice to seek
professional assessment if tightness becomes painful, persistent, or worse.
Root cause: the final synthesis did not preserve the case's non-diagnostic and
escalation boundaries even though the tool evidence was available.

### Evaluation Improvements

1. Add deterministic checks for declared tool argument values, semantic query
   terms, maximum call counts, and expected error/no-data results.
2. Add citation-completeness scoring for inline chunk IDs, separate from source
   retrieval and faithfulness.
3. Run judge-scored cases multiple times and report score distributions to make
   model-grader variance visible.
4. Calibrate criteria and faithfulness judgments against a small human-labeled
   set before using judge scores as hard regression gates.

## Interpreting Failures

A case passes only when all metrics applied to that case pass. This deliberately
separates three questions:

1. Did deterministic processing expose the expected facts?
2. Did the generated answer actually communicate the case requirements?
3. Were RAG and agent claims faithful to the evidence they received?

For example, `expected_data_points` can pass while `criteria_satisfaction`
fails. That means the system computed the expected values, but the generated
answer omitted or mischaracterized them. The evaluator records that distinction
without retrying generation or changing production prompts.

LLM-judge scores can vary between runs. Deterministic metrics should be used for
exact regression gates; judge metrics are best tracked as trends and reviewed
alongside their stored rationales.
