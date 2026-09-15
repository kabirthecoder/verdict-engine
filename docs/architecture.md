# Architecture

## 1. The idea in one paragraph

A decision is a set of claims. A claim stands only if it is backed by evidence and survives
every attack made on it. verdict-engine is a system where LLM agents make claims, gather
evidence by calling real tools, attack each other's claims, and audit the evidence — all by
writing into one shared **argument graph**. The verdict is not produced by any agent; it is
*computed* from the graph by a formal argumentation semantics. The graph is the proof.

## 2. Why not a supervisor

Supervisor and pipeline architectures have three structural problems for high-stakes
decisions: the supervisor's first framing of the problem biases every downstream agent; agents
are rewarded for completing the plan, not for being right; and the output is prose, so there is
nothing to audit. The court architecture inverts each: there is no framing step (any agent may
open a claim), one role is rewarded only for disproving, and the output is a graph with
provenance on every edge.

## 3. The argument graph

Nodes:

| Node | What it is | Key fields |
|---|---|---|
| `Question` | The decision being made | text, domain, status, budget |
| `Claim` | A proposition an agent asserts | text, kind (from domain vocabulary), author, status |
| `Evidence` | A recorded tool call | tool, args, raw output, output hash, summary, author, timestamp |
| `Attack` | "Claim A undermines claim B" | attacker claim, target claim, author, rationale |
| `Support` | "Evidence E backs claim C" | evidence, claim, quoted span, author |

Rules the graph enforces, not the prompts:

- A claim with no `Support` edges is *unsupported* and cannot be accepted, whatever it says.
- `Evidence` is immutable and written only by the tool runtime, never by an agent. An agent
  can only *reference* evidence it caused to be produced.
- An `Attack` must itself be a claim (so it can be attacked back) and must carry support.
- Every node records `author` (agent role + run id, or a human user id) and is append-only.
  Nothing is ever edited or deleted; retractions are new attacks.

## 4. Computing the verdict

The graph induces an abstract argumentation framework (Dung 1995): claims are arguments, the
`Attack` edges are the attack relation. We compute the **grounded extension** — the unique,
most skeptical set of arguments that defend themselves against every attacker. Claims in the
extension are `accepted`; claims attacked by an accepted claim are `rejected`; everything else
is `undecided`. The grounded extension is chosen over preferred/stable semantics precisely
because it is unique and cautious: the system never has to choose between multiple
self-consistent worlds, and it says *undecided* rather than guessing.

Unsupported claims are removed before the computation (they cannot attack or be accepted).
Audit failures (see §6) are attacks authored by the auditor and enter the graph normally.

The engine runs in milliseconds on graphs of thousands of nodes, is pure Python, and is fully
unit-tested against textbook cases.

## 5. Agents are triggered, not orchestrated

There is no scheduler that decides "now call the reviewer." Instead the **court loop** repeatedly
inspects the graph and fires roles when their trigger condition holds:

| Role | Trigger | Tools | Writes |
|---|---|---|---|
| Claimant | Question has no claims, or a subquestion was opened | domain tools | claims + supports |
| Investigator | A claim exists with insufficient support | domain tools | evidence + supports (or an attack, if the evidence contradicts) |
| Adversary | A claim is currently `accepted` and has not been attacked by an adversary in this round | domain tools | attacking claims + supports |
| Auditor | A `Support` edge exists that has not been audited | none (reads raw evidence only) | attacks on claims whose support misquotes the evidence |

Each firing is one bounded ReAct-style episode: the agent receives the relevant subgraph,
calls tools (parallel calls allowed), and returns structured output validated against a schema.
Every tool call is recorded as `Evidence` by the runtime before the agent sees the result.

The loop stops at a **fixed point** (a full round in which no role fires or nothing new is
written) or when the budget (tokens, tool calls, wall-clock) is exhausted. Then the verdict is
computed and frozen with the graph. A question can be **reopened** later with new evidence; the
old verdict stays in history.

## 6. Auditing

The auditor never sees agent summaries — only the raw evidence payload and the quoted span in
the `Support` edge. If the span does not occur in the payload, or the claim's meaning is not
entailed by the span, the auditor posts an attack. This is what makes hallucinated citations
structurally impossible to accept: they lose their support and drop out of the framework.

## 7. Domains as plug-ins

A domain provides: a claim vocabulary (kinds with schemas, e.g. `vulnerable(package, version,
advisory)`, `reachable(advisory, path)`), a tool set (typed functions with Pydantic schemas and
provenance), role prompts, and an eval set of historical questions with known verdicts. The
engine, graph, roles and loop are shared. First domain: `supply_chain`.

## 8. Runtime

Python 3.12, Pydantic v2, SQLAlchemy (SQLite for trial, Postgres for real), any
OpenAI-compatible chat API with tool calling (Ollama locally, Groq/OpenRouter free tiers,
or paid providers). FastAPI service and GitHub App come after the CLI works end to end.

## 9. What "production" means here

Provenance on every evidence node, append-only graph, reproducible verdicts (same graph → same
verdict, always), budgets with hard stops, an eval suite of historical cases run in CI, and a
security model where agents have no write access to anything outside the graph.
