# AI_WORKFLOW.md
## 1. Stage: Research & Scoping

**Goal**: understand the requirements, plan the architecture, and produce
`PLAN.md` before writing code.

- **AI tool(s) used**: Claude (claude.ai)
- **What I used it for**:
  - Summarize the requirements
  - Recommend possible approaches, tools, options
  - Analyze the overall structure, timeline, breakdown tasks
- **Notes**:
  - Ask AI to provide the source of information, price comparison, references to double check and make the final decision

---

## 2. Stage: Project Scaffold & Setup

**Goal**: repo structure, FastAPI skeleton, Docker Compose, env config.

- **AI tool(s) used**: Codex (GPT 5.5)
- **What I used it for**:
  - Generating boilerplate FastAPI app structure
  - Writing docker-compose.yml for app + Chroma
- **Review/correction notes**:
  - Initially, the coding agent still config the .env variables inside logic files

---

## 3. Stage: Feature 1 — Fitness Knowledge RAG

**Goal**: ingestion, chunking, embeddings, retrieval, out-of-scope handling,
guardrails.

- **AI tool(s) used**: Codex (GPT 5.5)
- **What I used it for**:
  - Chunking logic, Chroma integration, prompt templates for grounded answers
  - Drafting the out-of-scope classifier
  - Drafting the guardrails keyword list / classifier prompt
- **Prompting approach**:
  - Iterative small prompts, step-by-step, break into smaller tasks.
- **AI output that was wrong/suboptimal**:
  - **What it produced**: For the chunking logic, it initially produced one large ingest.py file
  - **Why it was wrong**: All data ingestion logics are put into one file
  - **How I corrected it**: Ask the agent to sketch out the structure (with instruction prompt) first, then guide the agent follow the structure.
- **Guardrails reflection**:
  - Did AI tools help or hinder your thinking on what the system should refuse?
    - The AI recommends the types of users' queries, the key word could appear in these queries that the system should refuse.

---

## 4. Stage: Feature 2 — Workout History Analysis

**Goal**: data processing (unit normalization, trend detection, deload
detection, etc.), insight generation endpoint.

- **AI tool(s) used**: Codex (GPT 5.5), Claude (claude.ai), ChatGPT (Web search & Deep research)
- **What I used it for**:
  - Writing the trend-detection/aggregation functions
  - Designing the structured-summary format passed to the LLM
  - Edge case handling: zero-weight entries, unit mixing, sparse data
- **Rejected AI suggestion**:
  - **What it suggested**: Initially, the trend-detection functions are simple functions that are put into one large analysis file.
  - **Why I rejected it**: Functions are generated without exception handling, or with improper implementation, and some are being hallucinated by AI (contains useless functions with no meaning for analysis).
  - **What I did instead**: Verify with another AI Agent (Claude), research more approaches with web search and deep research on ChatGPT. Then adjust the PLAN.md for more detail description for this feature, then guide the AI with proper analysis structure and functions.

---

## 5. Stage: Feature 3 — Coach Assist Agent

**Goal**: tool definitions, orchestration loop, multi-source synthesis,
graceful degradation.

- **AI tool(s) used**: Codex (GPT 5.5)
- **What I used it for**:
  - Tool schema definitions, orchestration loop structure
  - Designing the "insufficient data" handling logic
- **AI output that was wrong/suboptimal**:
  - **What it produced**: The agent flow without guardrail
  - **Why it was wrong**: The flow initially use the response from tools to input to the Orchestrator, so it sometimes accidentally bypass the guardrail of the RAG, and still use the data from the analysis tool to answer the questions.
  - **How I corrected it**: Add the guardrail layer on the orchestrator also.

---

## 6. Stage: Feature 4 — Evaluation Pipeline

**Goal**: test set, metrics (rule-based + LLM-as-judge), eval runner, results
analysis.

- **AI tool(s) used**: Codex (GPT 5.5), Claude (claude.ai), ChatGPT
- **What I used it for**:
  - Drafting the test set questions from the sample workout data
  - Writing the LLM-judge rubric/prompt
  - Writing rule-based metric checks
- **AI output that was wrong/suboptimal**:
  - **What it produced**: The draft set of test questions.
  - **Why it was wrong**: The set of generated evaluation set do not contain enough information to evaluate the system.
  - **How I corrected it**: Re-design the evaluation models, use different AI Agents for a proper set of questions and expected answer/criteria.

---

## 7. Stage: Documentation (README, EVALUATION.md, this file)

- **AI tool(s) used**: Codex (GPT 5.5), Claude (claude.ai)
- **What I used it for**:
  - Drafting architecture diagram, cost estimates, API docs
- **Review/correction notes**:
  - Re-check the cost estimation, architecture diagram

---

## 8. Summary Table — AI Tools by Stage

| Stage | Tool(s) used | Primary purpose |
|---|---|---|
| Research & scoping | Claude | Requirements analysis, architecture options, task breakdown, and planning |
| Scaffold & setup | Codex | FastAPI boilerplate, repository structure, Docker Compose, and environment configuration |
| Feature 1 — RAG | Codex | Ingestion and chunking, Chroma integration, grounded prompts, and guardrails |
| Feature 2 — Analysis | Codex, Claude, ChatGPT | Trend analysis, summary design, edge-case research, and implementation review |
| Feature 3 — Agent | Codex | Tool schemas, orchestration flow, multi-source synthesis, and agent-level guardrails |
| Feature 4 — Evaluation | Codex, Claude, ChatGPT | Evaluation dataset, rule-based metrics, LLM judge rubric, and result analysis |
| Documentation | Codex, Claude | Architecture diagram, cost estimates, API documentation, and document review |

---

## 9. Final Reflections

- **Where AI helped most**: AI was most useful for quickly producing project
  scaffolding, suggesting architecture and implementation options, and drafting
  repetitive code such as schemas, API endpoints, tests, and documentation. It
  also helped identify edge cases and compare alternative approaches during
  implementation.
- **Where AI helped least / got in the way**: AI was less reliable when prompts
  were too broad or when domain-specific reasoning was required. Some initial
  outputs placed too much logic in a single file, suggested functions without a
  clear analytical purpose, or produced evaluation questions that were not
  detailed enough. These outputs required manual review, external research, and
  verification with other AI tools.
- **What you'd do differently next time**: I would define module boundaries,
  expected behavior, edge cases, and acceptance criteria before asking AI to
  implement a feature. I would continue using small, iterative prompts, but add
  tests and verification criteria earlier. For important design or domain
  decisions, I would compare multiple sources and treat AI output as a draft
  rather than a final answer.
