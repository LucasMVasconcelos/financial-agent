# Financial AI Agent

A financial AI agent that talks over **Telegram** and recommends the next
best financial action (**Next Best Action — NBA**), built with **FastAPI**,
**LangChain**, and **Pydantic v2**.

## Architecture

> **Note on this branch (`feature/langgraph`)**: the diagram and table
> below already reflect this branch's state, where the main agent is also
> a LangGraph graph (`agent/main_graph.py`), no longer an `AgentExecutor`.
> See "Main graph (`feature/langgraph`)" below for why, and for what
> changes relative to `master`.

```
Telegram → FastAPI webhook → validation (secret token + rate limit)
         → complexity classification → Model Router (reasoning | utility tier)
         → MainGraph (LangGraph, cyclic FSM with a ReAct loop — agent/main_graph.py)
              reason ⇄ validate_tool_calls → execute_tool ⇄ self_correct
              ├─ get_customer_profile   → CustomerService      → CustomerRepository
              ├─ get_next_best_action   → NBAService           → NBAModelGateway (mock | SageMaker)
              ├─ get_products           → ProductsService      → CustomerRepository
              ├─ search_knowledge_base  → KnowledgeBaseService → KnowledgeBaseGateway (RAG, vector store)
              └─ request_loan           → LoanService          → LoanGraph (LangGraph, checkpointed)
                                                                      ├─ amount ≤ threshold → auto-approves
                                                                      └─ amount > threshold → pauses (PENDING_APPROVAL)
                                                                            → admin approves/rejects → resumes → notifies customer
         → natural-language response → Telegram

  MainGraph and LoanGraph share the same process-wide checkpointer
  (local MemorySaver | RedisSaver with USE_REDIS=true — see dedicated section).
```

Layers (`src/financial_agent/`):

| Layer | Folder | Responsibility |
|---|---|---|
| Domain | `domain/` | Pydantic models and the structured-error contract. No I/O. |
| Repository | `repositories/` | Abstracts where data lives (today: seeded in-memory fakes). |
| Gateway | `gateways/` | External integrations (Telegram Bot API, NBA model, vector store/RAG). |
| Service | `services/` | Use-case orchestration; translates failures into `ToolError`. |
| Agent | `agent/` | Prompt, Tools, Output Parser, the main graph (`main_graph.py`, LangGraph), and the loan flow (`loan_graph.py`, LangGraph). |
| Security | `security/` | Telegram webhook validation, service auth, rate limiting. |
| Observability | `observability/` | Structured logs (structlog), correlation id, tracing. |
| API | `api/` | FastAPI: routers, middleware, dependency injection. |
| Infra | `infra/aws/` | AWS Lambda adapter (Mangum). |

Every arrow above is a single-direction dependency (routers → services →
repositories/gateways), never the reverse — that is what lets any layer's
implementation be swapped (e.g. in-memory repository → Postgres, or the
mocked NBA model → SageMaker) without touching the others.

### Architectural decisions

- **Repository Pattern** for `CustomerRepository`/`ConversationRepository`:
  seeded in-memory fakes today; swapping in Postgres/DynamoDB only means a
  new class satisfying the same `Protocol`.
- **Gateway Layer** for everything external (Telegram, the NBA model):
  isolates latency, timeouts, and third-party payload formats from the
  rest of the code.
- **Service Layer**: the single place that decides "this is a `NOT_FOUND`"
  vs. "this is an `UPSTREAM_ERROR`" — Tools and any future REST endpoints
  reuse the same error translation.
- **Composition Root** (`api/app_state.py`): the only module that knows
  about concrete implementations; everything else depends on `Protocol`s.
- **Identity-required dispatch**: `user_id` is never a field on a Tool's
  Input Schema — it is injected via closure from the authenticated
  Telegram payload (`message.from.id`), never from the conversation text
  or a parameter the LLM could fill in.

## Main graph (`feature/langgraph`) — the whole project on LangGraph

On `master`, only the loan flow was a graph — the main conversational
agent was a LangChain `AgentExecutor` (an opaque ReAct loop: decide, act,
observe, repeat, with no states or transitions visible from the outside).
This branch replaces that with `agent/main_graph.py`: an explicit, cyclic
`StateGraph`, with every step of the ReAct loop as its own node,
checkpointed and resumable — fulfilling the request to migrate the whole
project to LangGraph, with an FSM, deterministic validation between steps,
retries, and checkpointing in Redis.

```
        START
          │
          ▼
    ┌─────────┐   no tool_calls, or             ┌──────────┐
 ┌─▶│  reason │──────iteration cap──────────────▶│ finalize │──▶ END
 │  └────┬────┘                                  └──────────┘
 │       │ tool_calls
 │       ▼
 │  ┌──────────────────┐   invalid      ┌───────────────┐
 │  │ validate_tool_    │───────────────▶│ self_correct  │
 │  │ calls             │                └───────┬───────┘
 │  └────────┬──────────┘                        │
 │           │ valid                              │
 │           ▼                                    │
 │     ┌─────────────┐   tool failed (ToolEnvelope│
 │     │ execute_tool│───success=false)───────────┘
 │     └──────┬──────┘
 │            │ success
 └────────────┘
```

- **`reason`** — the actual ReAct step: the LLM (already with
  `bind_tools`) sees the running message list (system prompt, history,
  prior tool results) and either calls a tool or produces the final
  answer. It is visited on every loop iteration — that is what makes the
  graph cyclic instead of a fixed, single-pass pipeline.
- **`validate_tool_calls`** — deterministic validation *between* steps,
  before any tool runs: an unknown tool name, an identity field
  (`user_id`) injected into the arguments (defense in depth — the Input
  Schemas already exclude that field; this is a second barrier in case a
  future schema regresses), or a loan amount out of the allowed range. A
  rejection here never reaches a service — it goes straight to
  `self_correct` with a synthetic `ToolEnvelope` error, in the exact same
  shape a real tool failure would produce, so the LLM can't tell the two
  apart.
- **`execute_tool`** — runs the already-validated tool call(s). No
  exception escapes here (contract of `run_tool`, see "Tools — contract"
  below); every result is a `ToolEnvelope` JSON string.
- **`self_correct`** — **Tool Self-Correction**: on seeing a structured
  error (validation failure or `ToolEnvelope.success=False`), it injects a
  corrective instruction into the conversation and loops back to `reason`
  instead of stalling the turn or giving up. Bounded by
  `_MAX_TOOL_RETRIES` (2) — after that, `route_after_execute` stops
  sending it back to `self_correct`; the LLM sees the failure one more
  time and decides how to close out (usually admitting it couldn't
  complete the action).
- **`finalize`** — extracts the final text once `reason` produces a
  response with no tool calls, or once the iteration cap
  (`_MAX_ITERATIONS = 6`, same as the `AgentExecutor`'s `max_iterations`
  on master) is hit.

### Human-in-the-loop: why it stays in `LoanGraph`, not the main graph

The original request reads "this branch uses LangGraph everywhere, with
the ability to pause for human validation" as if it were a single property
of a single graph. In practice that becomes **two coordinated LangGraph
graphs**, each solving the kind of pause that makes sense for it:

- `main_graph.py` **never** pauses a conversation turn waiting on a human.
  Pausing an entire turn for hours would be poor UX (the customer would
  keep seeing "typing..." indefinitely) and unnecessary — this graph's
  checkpoint only needs to survive one Telegram round-trip, not an
  indeterminate wait.
- `agent/loan_graph.py` is the right place for that pause: `request_loan`
  stays a regular Tool that returns immediately (with a `pending_approval`
  status when applicable), and it's the `LoanGraph` behind it that
  actually pauses at `await_decision` and resumes later via
  `POST /admin/loans/{id}/decide` — see "Loans" below.

The two graphs now share the **same process-wide checkpointer** (injected
in `api/app_state.py`, see next section) — thread_ids never collide
between them (`telegram:{update_id}` vs. `application_id`), so sharing one
instance is safe and avoids holding two redundant Redis connections.

### State persistence in Redis

`Settings.use_redis` (env `USE_REDIS`, the same flag that already selects
the rate limiter's backend) decides the checkpointer:

- `USE_REDIS=false` (default, and what the test suite uses) — in-process
  `MemorySaver`, lost on restart. Enough for local dev without Docker.
- `USE_REDIS=true` — `AsyncRedisSaver` (`langgraph-checkpoint-redis`),
  connected via `REDIS_URL`, with `asetup()` called once at startup
  (`api/app_state.py::_build_checkpointer`). **Requires Redis Stack**
  (`redis/redis-stack-server`, not `redis:7-alpine`) — the checkpointer
  indexes checkpoints via RediSearch (`FT.*` commands), which plain Redis
  doesn't have; `docker-compose.yml` already uses the right image on this
  branch. Confirmed by trying `AsyncRedisSaver` against a plain Redis:
  fails immediately with `unknown command 'FT._LIST'`.

With the checkpoint in Redis, a process restart mid-turn (between
`execute_tool` and `reason`, for instance) resumes from the last completed
node instead of losing the turn — the same cold-recovery reasoning that
already applied to `LoanGraph` on master, now extended to the main graph.

## LangChain — components used

| Component | Where | Role |
|---|---|---|
| Chat Model | `agent/llm_factory.py` + `agent/model_router.py` | `ChatOpenAI`, built in two variants (reasoning/utility) per process — see "Model Router" below. |
| System Prompt / Human Prompt | `agent/prompts/system_prompt_v4.py` (registered in `prompt_registry.py`), `agent/main_graph.py` | Persona, guardrails, and compliance versioned as code; `ChatPromptTemplate` composes system + history + human message (used only to *format* the graph's initial messages, not to run a loop). |
| Output Parser | `agent/output_parser.py` | `PydanticOutputParser` structures the "filler agent"'s response; the main graph reads `AIMessage.tool_calls` natively (`llm.bind_tools`), with no separate parser. |
| Tool Calling | `agent/tools/*.py` | Five `StructuredTool`s with a narrow schema and identity bound via closure — four read-only, one (`request_loan`) with a real effect. |
| Runnable | everywhere | Prompt, LLM, parser, Tools, and the compiled graphs themselves are all `Runnable`s composable with `|`. |
| LangGraph (`StateGraph` + checkpointer) | `agent/main_graph.py` | Main graph: cyclic FSM (`reason ⇄ validate_tool_calls ⇄ execute_tool ⇄ self_correct`) that replaces the `AgentExecutor` — see "Main graph" above. |
| LangGraph (`StateGraph` + checkpointer) | `agent/loan_graph.py` | A second, separate graph, Plan-and-Execute style, just for loan origination — see "Loans" below for why it's a graph separate from the main one. |

## Tools — contract

Every Tool (`agent/tools/*.py`) follows: narrow input schema → identity via
closure (never via the LLM) → handler → `run_tool` (`agent/tools/base.py`)
catches any exception and always returns a serialized `ToolEnvelope`:

```json
{"success": true, "data": {"...": "..."}, "error": null}
{"success": false, "data": null, "error": {"code": "NOT_FOUND", "message": "...", "details": {}}}
```

Error codes: `RATE_LIMITED`, `UPSTREAM_ERROR`, `NOT_FOUND`, `UNAUTHORIZED`,
`VALIDATION_ERROR`, `UNKNOWN_ERROR` — a raw exception never reaches the
agent loop.

## RAG — knowledge-base search

Of the four Tools, `search_knowledge_base` is the only one doing real
Retrieval-Augmented Generation — the other three do structured *lookups*
(function calls with fixed schemas), not semantic search.

```
search_knowledge_base(query)
    → KnowledgeBaseService (validates the query)
        → KnowledgeBaseGateway (Protocol)
            → InMemoryKnowledgeBaseGateway
                → InMemoryVectorStore (langchain_core) + OpenAIEmbeddings
                    → static corpus of ~8 articles (gateways/knowledge_base_gateway.py)
```

- The corpus (policies/how-it-works for CDs, treasury bonds, insurance,
  credit portability, early installment payoff, etc.) is **embedded once
  at startup** (`InMemoryKnowledgeBaseGateway.build`, called in
  `api/app_state.py`), not on every request.
- `query` is this Tool's only Input Schema field, and the one case where
  we let the LLM freely control a parameter — because it carries no
  identity and doesn't filter another customer's data, it only
  parameterizes *what* is searched for.
- System Prompt v2 instructs the agent to answer **only** based on what
  the tool returned, never filling gaps with the model's own knowledge —
  and to never use this tool in place of `get_next_best_action` for
  personalized recommendations.
- Swapping the backend: `InMemoryVectorStore` is fine for a dozen static
  articles; for production/scale, implement a new class of the
  `KnowledgeBaseGateway` Protocol over a real vector store (pgvector,
  Pinecone, OpenSearch, ...) — no other layer changes.
- Tests never call the real embeddings API: they use
  `DeterministicFakeEmbedding` (`langchain_core`), deterministic and
  network-free (see `tests/conftest.py` and
  `tests/unit/test_knowledge_base_gateway.py`).

## Model Router — smaller vs. larger model, by activity and by complexity

`agent/model_router.py` holds two `ChatOpenAI` clients, built once at
startup, and routes along **two independent axes**:

**By activity** (`for_activity`, fixed per call site):

| Activity | Tier | Why |
|---|---|---|
| `AGENT_REASONING` | **reasoning** (`OPENAI_REASONING_MODEL`, default `gpt-4o`) | Decides which tool to call, follows the compliance guardrails, writes the final answer. |
| `FILLER_REPLY` (holding message) | **utility** (`OPENAI_UTILITY_MODEL`, default `gpt-4o-mini`) | Short sentence, no tools, no compliance guardrails — latency matters more than quality here. |
| `SUMMARIZATION` (history compression) | **utility** | Text compression is a mechanical task, not judgment. |

The filler **is only sent if the main agent hasn't replied yet after
`FILLER_DELAY_SECONDS`** (default 2.5s) — `telegram_webhook.py` runs the
agent as a background task and only fires the filler if it hasn't finished
within that window. Sending the filler unconditionally would make the
holding message land right behind the real answer on any fast turn (most
of them), reading as two replies/spam instead of latency-hiding UX.

**By message complexity** (`for_complexity`, decided per turn by the
webhook): `agent/query_complexity.py::classify_query_complexity` classifies
each message as `SIMPLE` or `COMPLEX` with a **deterministic heuristic
that makes no LLM call** (message length, comparison/justification signal
words, loan mentions, multiple questions in the same text) — spending a
model call just to decide which model to use would cancel out the gain the
routing exists to provide. `SIMPLE` uses the utility tier, `COMPLEX` uses
the reasoning tier — it's this choice, not a fixed `AGENT_REASONING`, that
decides the main graph's model on every turn (`telegram_webhook.py`).

### ReAct on the small model, Plan-and-Execute (LangGraph) for the complex flow

The intuitive pairing ("bigger model = fancier technique") is deliberately
not what's implemented. Simple and complex questions go through the
**same** graph/ReAct loop (`reason ⇄ validate_tool_calls ⇄ execute_tool`,
see "Main graph" above) — only the model size underneath changes, because
deciding "call one tool or answer" doesn't change shape with how complex
the question is. Where the technique actually changes is the loan
request: being the one action with a real effect, always handled with the
reasoning tier, and needing a revisable plan with a pause point for human
approval — that's where `LoanGraph` (LangGraph, Plan-and-Execute style, a
graph separate from the main one) comes in, instead of one more regular
Tool inside the ReAct loop. See "Loans" below for the full reasoning.

## Loans — the one action with a real effect (LangGraph + human-in-the-loop)

Of the five Tools, `request_loan` is the only one that actually changes
state — the other four are read-only. That's why it isn't just "one more
tool" that the main graph's `execute_tool` calls and moves on: behind it
sits a second, dedicated `LangGraph` graph (`agent/loan_graph.py`), because
the problem it solves — "may need to pause for an indeterminate amount of
time waiting on a human to decide, and resume exactly where it left off,
well after the Telegram request itself has already finished" — isn't
something a regular tool, executed within a single conversation turn, can
express (see "Human-in-the-loop" above for why it's a second graph instead
of the same main graph).

```
request_loan(amount)
  → LoanService.request_loan → LoanGraph.ainvoke(initial_state, thread_id=application_id)

        assess (queries CustomerService)
            │
        route_by_amount
            ├─ amount ≤ LOAN_HUMAN_APPROVAL_THRESHOLD (default R$ 50,000)
            │     → auto_approve → END                                   [status: approved]
            │
            └─ amount > threshold
                  → await_decision → END                                 [status: pending_approval, PAUSES here]

  (later, outside any Telegram request)
  POST /admin/loans/{id}/decide  (auth: X-Service-Api-Key)
        → LoanService.decide
              → graph.aupdate_state(thread_id, {human_decision: "approved"|"rejected"})
              → graph.ainvoke(None, thread_id)   # resumes from the checkpoint
                    → route_by_decision → finalize → END      [status: disbursed | rejected]
        → notifies the customer via Telegram (best-effort)
```

- **Checkpointer**: the same process-wide checkpointer as the main graph
  (`MemorySaver` by default, `RedisSaver` with `USE_REDIS=true` — see
  "State persistence in Redis" above); before this branch, each graph had
  its own isolated `MemorySaver`.
- **Why two `ainvoke`s instead of one blocking call**: the first one runs
  inside the Telegram request and must return fast — it never waits on a
  human. The second happens minutes, hours, or days later, triggered by
  the admin endpoint, completely outside that conversation's lifecycle.
- **`LoanRepository`** (`repositories/loan_repository.py`) exists
  *alongside* the graph's checkpointer: the checkpointer is indexed by
  `thread_id` and wasn't built to be listed ("show all pending
  applications"); the repository is the clean, queryable view the admin
  endpoint uses.
- **The most important guardrail in prompt v4**: `requires_human_approval=True`
  means pending, not "basically approved" — the agent is instructed to
  never state a loan was approved/disbursed unless the `status` field
  returned by the tool says so explicitly.
- **Customer notification is best-effort**: if delivery over Telegram
  fails after the human decision, the decision is already persisted — the
  customer just misses the immediate notice, they're never left with an
  inconsistent state.

### Testing the flow (with the already-seeded users)

| `user_id` | Test amount | Expected path |
|---|---|---|
| `123` (Ana Souza) | `R$ 10,000` | Auto-approved — `status: approved` |
| `456` (Bruno Lima) | `R$ 15,000` | Auto-approved — `status: approved` |
| `123` or `456` | `R$ 80,000` | Pending — `status: pending_approval`, shows up in `GET /admin/loans` |

To decide a pending application (replace `<id>` with the `application_id`
returned by the conversation or by `GET /admin/loans`):

```bash
curl -X POST "http://localhost:8000/admin/loans/<id>/decide" \
  -H "X-Service-Api-Key: <SERVICE_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "decided_by": "ana.analista"}'
```

## Memory: short term, summary, and long term

Three distinct mechanisms, with different lifecycles (see
`domain/models/conversation.py` for why they live in separate fields
instead of a single list):

1. **Raw window** (`ConversationHistory.messages`) — the current
   conversation's most recent messages, loaded via
   `ConversationRepository`.
2. **Rolling summary** (`ConversationHistory.summary`) — once the raw
   window passes 20 messages, `ConversationService._maybe_compact` merges
   the 10 oldest into a summary (`agent/conversation_summarizer.py`, the
   router's utility model) and removes them from raw storage via
   `ConversationRepository.compact`. Replaces the hard cutoff that used to
   exist — nothing is simply discarded, it's compressed.
3. **Long-term semantic memory** (`ConversationHistory.long_term_memories`)
   — every time a summary is produced, it also becomes an entry in a
   per-user vector store (`gateways/semantic_memory_gateway.py`, an
   `InMemoryVectorStore` isolated by `user_id` — no filter, structural
   isolation). At the start of every turn, `ConversationService.get_history`
   searches these memories semantically using the current message as the
   query, surfacing relevant context even after it has "scrolled" out of
   both the raw window and the summary — memory that spans conversations,
   not just turns.

Both ends of long-term memory (`SemanticMemoryService.remember` and
`.recall`) and compaction (`ConversationService._maybe_compact`) are
**best-effort**: a failure in the embeddings backend or the summarization
model is logged and swallowed — it never blocks delivering the reply to
the customer, which has already been computed by that point in the flow.

System Prompt v3 receives `{conversation_summary}` and
`{long_term_memories}` as template variables and instructs the agent to
treat them as approximate context, never as exact fact — for current
balance, products, or recommendations, the agent always confirms via a
tool, never trusts memory alone.

## Security

- **Telegram webhook**: authenticated via the
  `X-Telegram-Bot-Api-Secret-Token` header (configured in `setWebhook`),
  compared in constant time — Telegram doesn't sign the payload by
  default, so this is the recommended mechanism.
- **user_id**: always read from `message.from.id` (Telegram's own
  authenticated field), never from the message text or a Tool parameter.
- **Rate limiting**: per `user_id`, before any LLM call (fixed window,
  in-memory or Redis).
- **Service auth**: internal endpoints (outside the webhook) require
  `X-Service-Api-Key` — this includes the loan admin endpoints
  (`api/routers/admin_loans.py`), no exceptions.
- **`request_loan` is the only Tool with a real effect**: the other four
  are all read-only. The amount (`amount`) is the only parameter the LLM
  freely controls on it — same logic as `query` on
  `search_knowledge_base`: it carries no identity, it only parameterizes
  the authenticated customer's own request.

## Observability

- Structured logs (`structlog`), JSON in production/staging, readable
  console output in dev.
- `correlation_id`/`request_id` per request (`CorrelationIdMiddleware`),
  automatically merged into every log line.
- `traced_span` (`observability/tracing.py`) measures and logs the
  duration of every Tool call; emits real OpenTelemetry spans if the
  optional SDK is installed and configured.
- **LangSmith** (`LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY`) for
  end-to-end LLM tracing — chosen over LangFuse because a native hook
  already exists (`agent/llm_factory.py::configure_langsmith_tracing`) and
  because it's LangChain's own managed service, requiring no new
  infrastructure in `docker-compose.yml`. Just set the two variables; no
  code changes. `LoanGraph` never calls an LLM (it's a deterministic state
  machine), so there's nothing of it for LangSmith to trace beyond what
  the structured logs already cover — only the main graph's (`reason`)
  calls against `ChatOpenAI` show up in the trace.

## Running locally with Docker Compose

1. Copy the example file and fill in the secrets:

   ```bash
   cp .env.example .env
   ```

   Fill in `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET` (a random string
   of your choosing), `OPENAI_API_KEY`, `SERVICE_API_KEY`,
   `NGROK_AUTHTOKEN`, and `NGROK_DOMAIN` (see "Development tunnel" below).

2. Bring up the services:

   ```bash
   docker compose up --build
   ```

   The API comes up at `http://localhost:8000`; `GET /health` should
   respond `{"status": "ok"}`. The compose `ngrok` service already exposes
   that port publicly — no need to run a separate tunnel.

### Development tunnel (fixed domain)

Telegram needs to reach the API over public HTTPS, and the webhook needs
to survive restarts. An ephemeral tunnel (a bare `ngrok http 8000`, or
`trycloudflare.com`) draws a **new URL on every restart**, which silently
invalidates the previous `setWebhook` — the symptom is "I send a message
and nothing happens," with `getWebhookInfo` showing `last_error_message`
and pending updates. That's why `docker-compose.yml` includes a dedicated
`ngrok` service with a **reserved static domain**, so the URL never
changes across restarts:

1. Create a free account at [ngrok.com](https://ngrok.com) and get the
   authtoken at
   [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
   → `NGROK_AUTHTOKEN`.
2. Reserve a free static domain at
   [dashboard.ngrok.com/domains](https://dashboard.ngrok.com/domains)
   (e.g. `your-name.ngrok-free.app`) → `NGROK_DOMAIN`.
3. `docker compose up` already brings up the tunnel; check the active
   public URL (should match `NGROK_DOMAIN`) in the local dashboard at
   `http://localhost:4040`.

With a fixed domain, the next step's `setWebhook` only needs to be done
**once** — restarting `docker compose` no longer breaks the webhook.

### Configuring the Telegram bot

1. Create the bot with [@BotFather](https://t.me/BotFather) and get the
   token (`TELEGRAM_BOT_TOKEN`).
2. Pick a random secret for `TELEGRAM_WEBHOOK_SECRET` (e.g.
   `openssl rand -hex 32`).
3. Register the webhook, pointing at the tunnel's fixed domain:

   ```bash
   curl -X POST "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook" \
     -H "Content-Type: application/json" \
     -d '{
           "url": "https://<NGROK_DOMAIN>/webhook/telegram",
           "secret_token": "<TELEGRAM_WEBHOOK_SECRET>"
         }'
   ```

   To debug (see the current URL, delivery errors, pending updates):

   ```bash
   curl "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getWebhookInfo"
   ```

4. Chat with the bot on Telegram. The seeded example users
   (`repositories/customer_repository.py`) are `123` (Ana Souza) and `456`
   (Bruno Lima) — to test with a real Telegram `user_id`, add your own id
   to the in-memory repository.

## Running without Docker (development)

```bash
poetry install
poetry run pre-commit install
cp .env.example .env  # fill in the values
poetry run uvicorn financial_agent.main:app --reload
```

## Tests, lint, and type-check

```bash
poetry run pytest              # unit + contract + api tests, with coverage
poetry run ruff check .        # lint
poetry run ruff format .       # formatting
poetry run mypy src            # strict type-check
poetry run pre-commit run --all-files
```

## Golden Transcripts — agent behavior regression

`pytest` alone doesn't catch everything: the tests above never call a real
LLM (fakes, `DeterministicFakeEmbedding`, mocked `run_agent_turn` in the
API tests) — on purpose, to keep the suite fast, deterministic, and free
to run on every commit. That leaves a real gap: nothing guarantees a
prompt change, a Tool description change, or a model swap hasn't broken
the agent's *behavior* — only that the code still runs. **Golden
Transcripts** cover that gap: conversation scenarios with behavioral
assertions (which tools should or shouldn't be called, what the final
answer should/shouldn't contain), meant to run against the real agent.

```
tests/golden/
├── schema.py                     # GoldenTranscript, ExpectedToolCall, ResponseAssertions, rubric (pydantic)
├── judge.py                      # LLM-as-a-judge: evaluates each scenario's rubric field
├── transcripts.yaml              # the scenarios — 12 examples covering the 5 tools + guardrails + security
├── test_golden_transcripts_schema.py   # validates the YAML (no LLM, runs in the normal pytest suite)
└── run_golden_transcripts.py     # actually runs against the agent (needs OPENAI_API_KEY, manual)
```

- **`response_assertions` is substring-based, never exact-text** — the
  same question produces different phrasing across runs, even at low
  temperature. What needs to stay constant is the *content*, not the
  wording: for example, `loan_large_amount_requires_human_approval` (the
  most important scenario in the file) checks that the response never
  contains "approved"/"disbursed" while the loan is pending human
  review — that's a compliance regression if it fails, not just a
  wording-quality one.
- **`test_golden_transcripts_schema.py` runs in normal CI** (no network)
  and catches authoring mistakes — a tool name that no longer exists, a
  duplicate id, a category with no example — before even spending an API
  call.
- **Actually running against the agent is manual**, by design — it costs
  money and isn't deterministic:

  ```bash
  OPENAI_API_KEY=sk-... poetry run python tests/golden/run_golden_transcripts.py
  OPENAI_API_KEY=sk-... poetry run python tests/golden/run_golden_transcripts.py --id loan_large_amount_requires_human_approval
  ```

- **Adding a new scenario**: one entry in `transcripts.yaml`, no Python
  code to touch — the schema validates it automatically. Run `pytest
  tests/golden/` afterward to confirm the YAML is well-formed.

### LLM-as-a-judge — the third layer of assertion

`expected_tool_calls` and `response_assertions` are fast, deterministic,
and free — but structurally blind to qualitative quality: appropriate
tone, whether an explanation makes sense, whether the response is
*faithful* to what a tool returned (not just mentions the right word).
That's what the optional `rubric` field on `GoldenTranscript` is for,
evaluated by `judge.py` (`GoldenTranscriptJudge`) — a second LLM call,
separate from the conversation itself, that judges the response against
**one** free-text criterion and returns a structured `{passed: bool,
reasoning: str}`.

- **Additive, not a replacement**: the deterministic checks remain the
  first line of defense, always. The judge only kicks in when `rubric` is
  set, as a second opinion where substring matching can't reach —  e.g.
  in `guardrail_out_of_scope_legal_advice` (where the note used to say
  "better evaluated by human review") and in
  `loan_large_amount_requires_human_approval`, to catch the case where the
  response dodges the forbidden words but still *implies* approval.
- **Model Router's reasoning tier** (`ModelActivity.JUDGE`) — judging tone
  and faithfulness is real judgment, not a mechanical task; the same
  cost/quality reasoning that governs the rest of the routing.
- **A deliberate scope choice**: implemented as an extension of the
  existing `run_golden_transcripts.py`, not as a LangSmith
  Datasets/evaluators integration — needs no external account/setup to be
  useful right now. Migrating the scenarios to a LangSmith Dataset and
  running them via `evaluate()` remains a natural next step later, if you
  want a dashboard and trend history instead of a terminal report.

## Replacing the mocked model with a real one

The entire swap happens at a single extension point:
`gateways/nba_model_gateway.py`, behind the `NBAModelGateway` Protocol
(`async def predict(customer: CustomerProfile) -> NextBestActionCandidate`).

1. Implement a new class satisfying that Protocol (e.g. calling a REST
   endpoint, a local model, or the already-included
   `SageMakerNBAModelGateway` as a reference).
2. Register it in `api/app_state.py::_build_nba_model_gateway`, gated by a
   new `Settings.nba_model_provider` option.
3. No other layer changes: `NBAService`, the `get_next_best_action` Tool,
   the prompt, and the contract tests all remain valid, since they only
   depend on the Protocol.

## AWS (SageMaker / Lambda)

- **A real model via SageMaker**: implement/adjust
  `gateways/nba_model_gateway.SageMakerNBAModelGateway` (already included)
  to match your endpoint's payload format, set `NBA_MODEL_PROVIDER=sagemaker`,
  `SAGEMAKER_ENDPOINT_NAME`, and `AWS_REGION`, and install the optional
  dependency group: `poetry install --with aws`.
- **The API via AWS Lambda**: `infra/aws/lambda_handler.py` exposes
  `handler`, adapting the same FastAPI app via [Mangum](https://mangum.io/)
  — no route or service needs to change. Package it as a container image
  (reuse the `Dockerfile`, swapping the `CMD`) or as a dependency zip,
  publish it behind API Gateway, and point Telegram's `setWebhook` at the
  API Gateway URL. Configure the same environment variables from
  `.env.example` as the Lambda function's environment variables.
- Cold starts: keep the function warm (provisioned concurrency) if webhook
  latency is critical — Telegram expects a response within a few seconds.
