# AI Workout Coach — Implementation Plan

## Overview
Build an AI Workout Coach with four components: a fitness RAG pipeline, a workout
history analysis endpoint, a coach-assist agent that orchestrates both, and an
evaluation pipeline. Model, embedding, retrieval, and vector-store integrations
use **LangChain** with the **OpenAI API** and **Chroma**. The service is exposed
via **FastAPI** and runs locally via
`docker compose up`.

## Tech Stack
- Language: Python, FastAPI
- AI framework: LangChain (`langchain-openai`, `langchain-chroma`)
- LLM: OpenAI (e.g. `gpt-4o-mini` for retrieval/analysis generation, `gpt-4o` for
  agent orchestration and LLM-judge evaluation) — model names configurable via env vars
- Embeddings: OpenAI `text-embedding-3-small`
- Vector DB: Chroma (local, persisted via volume)
- Config: all secrets/keys via environment variables (`.env`, never hardcoded)

## Repository Structure
```
/app
  /core
    ai.py            - shared LangChain OpenAI chat + embedding factories
    chroma.py        - shared Chroma HTTP client factory
  /rag
    models.py        - ingestion data models
    documents.py     - knowledge-base file discovery
    tokenization.py  - embedding-model token counting and hard splits
    chunking.py      - header-aware Markdown chunk construction
    vector_store.py  - LangChain Chroma creation, upserts, stale cleanup
    ingest.py        - ingestion orchestration + CLI
    retrieve.py       - top-k retrieval + grounded prompt construction
    guardrails.py     - out-of-scope + medical/ED refusal classifier
  /analysis
    processing.py     - trend detection, volume aggregation, neglected-exercise logic
    insight.py        - builds structured summary -> LLM prompt -> insight
  /agent
    tools.py          - tool schemas + handlers (rag_search, analyze_history)
    orchestrator.py   - tool-calling loop, multi-source synthesis
  /api
    main.py           - FastAPI app, route registration
    routes_rag.py
    routes_analysis.py
    routes_agent.py
  /eval
    test_set.py       - QA pairs (RAG, analysis, agent, adversarial)
    metrics.py        - rule-based + LLM-judge metrics
    run_eval.py        - runner, outputs results
/data
  knowledge_base/      - provided markdown docs (+ any supplemental docs, documented)
  sample_workouts.json - provided sample workout history
docker-compose.yml
.env.example
README.md
AI_WORKFLOW.md
EVALUATION.md
PLAN.md
```

## Feature 1 — Fitness Knowledge RAG

### Ingestion (`rag/ingest.py`)
- Load all markdown files from `/data/knowledge_base`.
- Chunk header-aware: treat `##` sections as semantic units, split oversized
  sections at `###` or paragraph boundaries, and balance adjacent short sections
  into chunks targeting ~300 tokens (120-token soft minimum, 450-token hard
  maximum). Add token overlap only when a single atomic text block must be split;
  natural heading boundaries do not need overlap and avoiding it reduces duplicate
  retrieval results.
- Embed each chunk through LangChain `OpenAIEmbeddings`, configured with
  `text-embedding-3-small` by default.
- Store in Chroma with deterministic `chunk_id` values and metadata:
  `source_file`, `document_title`, `section_title`, `section_titles`,
  `primary_section_title`, `section_path`, `section_paths`, `chunk_index`,
  `token_count`, `embedding_model`, source/content hashes, and the ingestion
  owner marker. Multi-value metadata is JSON-encoded so it remains compatible
  with Chroma's scalar metadata values.
- Make ingestion idempotent with upserts and remove stale chunks previously owned
  by this ingestion pipeline after a successful embedding/upsert pass.
- Use the same shared LangChain provider factories for retrieval query embeddings,
  RAG chat generation, analysis generation, and agent model calls.
- If supplementing the knowledge base with extra docs, place them in the same
  folder and note the addition + rationale in README.

### Retrieval (`rag/retrieve.py`)
- Endpoint: `POST /rag/query` — input: `{ "question": str }`.
- Flow: embed query -> top-k similarity search in Chroma -> build a grounded
  prompt that instructs the LLM to answer ONLY from provided context and to
  cite which chunk(s) it used -> call OpenAI -> return `{ answer, sources: [...] }`
  where each source includes `source_file` and `section_title`.
- Out-of-scope handling: before retrieval, run a lightweight classification step
  (LLM call or rule-based topic check) to detect non-fitness queries (e.g. "what's
  the weather"). If out-of-scope, skip retrieval and return a polite message
  stating the assistant only handles fitness/training questions.

### Guardrails (`rag/guardrails.py`)
- Runs as a pre-check before the RAG flow generates an answer.
- **Trigger signals**: requests for diagnosis of pain/injury without professional
  assessment, eating-disorder-risk content (extreme calorie restriction, purging,
  "how to lose weight fast" combined with disordered-eating signals), explicit
  medical diagnosis requests ("do I have a torn rotator cuff?").
- **Implementation**: two-layer check — (1) keyword/pattern match against a curated
  list of medical/injury/ED red-flag terms, (2) LLM-based intent classifier as a
  fallback for phrasing that evades keywords.
- **Response on trigger**: a fixed, supportive refusal message that does not
  attempt diagnosis, recommends consulting a qualified professional (doctor,
  physical therapist, registered dietitian), and where appropriate offers general,
  non-diagnostic safety information (e.g. general RICE guidance framed as "general
  information, not medical advice").
- **Avoiding over-restriction**: guardrails must NOT fire on general technique,
  programming, nutrition-basics, or recovery questions. Keyword list should be
  scoped to injury/diagnosis/ED-specific phrasing, not broad fitness terms like
  "pain" alone — require co-occurrence with diagnostic/treatment-seeking intent
  (e.g. "is this a tear", "what's wrong with my knee", "how do I treat my
  injury") rather than single keywords.
- Document the full refusal strategy (signals, message, anti-over-restriction
  approach) in README/EVALUATION as required.

## Feature 2 — Workout History Analysis

### Data processing (`analysis/processing.py`)
Pure-Python functions, no LLM involved, run before any prompt is built:
- **Trend detection**: per exercise, compute estimated 1RM or top-set weight over
  time, linear trend (slope), percent change over the requested window.
- **Volume aggregation**: total volume (sets x reps x weight) per exercise and per
  muscle group, using a small exercise -> muscle-group lookup table.
- **Neglected exercises**: detect exercises/muscle groups with low frequency or
  long gaps since last performed.
- **Overtraining ratio**: compare volume between muscle groups (e.g. chest vs
  back) over a time window.

### Insight generation (`analysis/insight.py`)
- Endpoint: `POST /analysis/query` — input: `{ "user_id": str, "history": [...],
  "question": str }`.
- Flow: classify question intent (trend / neglect / balance / plan-suggestion) ->
  run the relevant processing function(s) -> build a structured, numeric summary
  (dates, weights, percentages, deltas) -> pass ONLY this summary (never raw JSON)
  to the LLM -> LLM produces a natural-language insight referencing the specific
  numbers.
- **Edge cases**:
  - Empty history -> return a clear "not enough data to analyze" message, no LLM
    call needed.
  - Insufficient data for a trend (e.g. fewer than 2 sessions for an exercise) ->
    explicit caveat in the response.
  - Unknown/unrecognized exercise name -> flag it, optionally fuzzy-match to a
    known exercise, and note the assumption.
- **Data isolation**: workout history is scoped strictly to the `user_id` provided
  in the request; no cross-user data access at any layer. Include a test that
  confirms requesting analysis for User A cannot surface User B's data even under
  adversarial input (e.g. user_id mismatch, crafted questions referencing another
  user).

## Feature 3 — Coach Assist Agent

### Tools (`agent/tools.py`)
- `rag_search(query: str)` — wraps Feature 1's retrieval logic directly (not via
  HTTP), returns `{ answer, sources }`.
- `analyze_history(user_id: str, question: str)` — wraps Feature 2's analysis
  logic directly, returns the structured insight + underlying numeric summary.
- Both tools are defined as OpenAI function-calling tool schemas.

### Orchestration (`agent/orchestrator.py`)
- Endpoint: `POST /agent/query` — input: `{ "user_id": str, "question": str }`.
- Flow: send the question + tool schemas to the LLM -> LLM decides which tool(s)
  to call and in what order (no hardcoded sequence) -> execute requested tool
  call(s) -> feed results back to the LLM -> repeat until the LLM produces a final
  answer (bound the loop with a max-iterations safeguard).
- Final response must be a single coherent answer. If both tools were used, the
  response must cite both data sources (e.g. "based on your training history...
  and general guidance on progressive overload...").
- **Graceful degradation**: if a tool returns insufficient data (e.g. analysis
  says "not enough history"), the agent must acknowledge this limitation in its
  final answer rather than fabricating data-backed claims.

### Design notes to document in README
- What happens if the agent calls the wrong tool first: the loop allows the LLM
  to observe the (unhelpful) tool result and call a different/additional tool
  before finalizing — no rigid pipeline.
- Adding a third tool: register a new tool schema + handler function in
  `agent/tools.py`; no changes needed to the orchestration loop itself.
- Biggest production failure-mode concern: tool-call hallucination (LLM invents
  arguments not grounded in the request, e.g. wrong user_id) or unbounded
  tool-call loops increasing latency/cost — mitigated via the max-iterations cap
  and strict argument validation before executing a tool.

## Feature 4 — Evaluation Pipeline

### Test set (`eval/test_set.py`)
- 5 RAG QA pairs (question, expected grounded answer/topic, expected source doc).
- 5 workout-analysis QA pairs using the provided sample workout JSON (question,
  expected data points referenced).
- 3 agent multi-step questions (mirroring the examples in the brief).
- 2 adversarial guardrail cases: one injury/diagnosis-seeking question, one
  eating-disorder-risk question — both must trigger a refusal.

### Metrics (`eval/metrics.py`)
- **Rule-based — source attribution**: for RAG outputs, verify `sources` is
  non-empty and references a real document/chunk.
- **Rule-based — data grounding**: for analysis outputs, verify the response text
  contains numeric values/dates present in the computed summary (regex/number
  matching against the structured summary).
- **Rule-based — guardrail correctness**: verify refusal fires on the 2
  adversarial cases and does NOT fire on the legitimate fitness QA pairs.
- **LLM-as-judge — faithfulness**: for RAG and agent outputs, prompt a judge model
  to score (1-5) whether the answer is fully supported by the retrieved context /
  tool outputs, with a short rubric and rationale returned alongside the score.

### Runner (`eval/run_eval.py`)
- Iterates the full test set, calls the relevant endpoint/function for each case,
  applies all metrics, and writes results (per-case + aggregate) to a results
  file consumed by `EVALUATION.md`.

## Documentation Deliverables
- **README.md**: architecture overview (with diagram), setup/run instructions
  (`docker compose up`), API documentation for all endpoints, design decisions and
  tradeoffs, cost-per-query estimate for Features 1 & 2 at 1,000 queries/day and
  what to optimize first (e.g. caching, embedding reuse, smaller model for
  classification steps).
- **AI_WORKFLOW.md**: tools used per stage, examples of incorrect/suboptimal AI
  output and how it was corrected, at least one rejected AI suggestion and why,
  prompting strategy, and a reflection on the guardrails implementation.
- **EVALUATION.md**: full test set, results per metric, at least 2 failure
  examples with root-cause analysis, and improvement ideas for the next iteration.

## Bonus — Usage Metering (architecture only, no code)
Describe (≤200 words) how to add per-query metering: what to meter (token counts
and/or call counts per endpoint, tagged by coach/account ID), where to enforce
limits (middleware/decorator at the API gateway layer before the LLM call, with a
pre-flight quota check), and how to handle a coach hitting their limit mid-session
(graceful degradation — e.g. return a clear "limit reached" response instead of
calling the LLM, with an option to queue/upgrade rather than hard-failing).
