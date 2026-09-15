# verdict-engine

**Multi-agent decisions you can audit, attack and defend.**

Most multi-agent AI systems put one "supervisor" agent in charge. It picks a plan, the other
agents follow, and you get a confident answer with no way to check it. That is exactly why
companies don't trust AI with decisions that matter.

verdict-engine replaces the supervisor with a **court**.

- Agents don't pass messages to each other. They write **claims**, **evidence** and **attacks**
  into a shared **argument graph**.
- Every piece of evidence is the recorded result of a real **tool call** — the tool, the
  arguments, the raw output and its hash. No evidence, no claim.
- Some agents exist only to **attack**. Adding agents makes the answer more contested, not more
  agreeable.
- No agent decides. A formal **argumentation engine** (grounded semantics) reads the graph and
  returns which claims survive every attack. The graph is the proof.
- If nothing survives, the answer is *undecided* — with the exact unresolved argument shown.
- A human can join the court as another participant with the same standing as an agent.

The engine is domain-agnostic. A domain is a plug-in: a vocabulary of claims plus a set of
tools that produce evidence. The first domain is **software supply-chain trust**: is this
package safe to adopt, is this vulnerability reachable, is this contributor's PR safe to merge.

## Status

Early. Being built in the open, See `docs/architecture.md` for the design
and `docs/roadmap.md` for what's next.

## Quick start

```bash
uv sync --all-extras
cp .env.example .env         # point it at Ollama, Groq, OpenRouter or any OpenAI-compatible API
uv run verdict doctor        # can this machine reach the LLM, OSV, deps.dev, PyPI, npm, GitHub?
uv run verdict ask "Is requests 2.28.0 safe to adopt?"
uv run verdict ask "Is event-stream 3.3.6 safe to adopt?" -c ecosystem=npm
uv run verdict show <case id>          # the proof: claims, labels, attacks, audited citations
uv run verdict eval                    # historical incidents with known verdicts
```

Any OpenAI-compatible endpoint with tool calling works. Free options that handle
multi-step tool use well enough: `qwen3:14b` or larger on Ollama, `llama-3.3-70b-versatile`
on Groq's free tier, or the `:free` models on OpenRouter. Small 7-8B models will make
mistakes — that is fine, the adversary and auditor exist to catch them, and `verdict show`
lets you see exactly where.

## How a case runs

```
round 1   claimant      calls package_info, osv_query, project_info, repo_activity…
                        posts claims, each citing an evidence id + verbatim quote
          auditor       checks every quote against the raw tool output
          adversary     tries to break the accepted claims: transitive deps, wrong
                        version, misread advisory ranges… posts cited attacks
round 2   investigator  digs into any claim that is unsupported or under attack
          auditor       audits the new citations
          adversary     attacks whatever is accepted now
…         until a round writes nothing (fixed point) or the budget runs out
verdict   grounded semantics over the graph: accepted / rejected / undecided /
          unsupported — computed, not decided by any agent
```

## Layout

```
src/verdict/graph      argument graph models + append-only store (sqlite/postgres)
src/verdict/engine     grounded semantics → verdict labels
src/verdict/tools      @tool decorator, runtime that records every call as evidence
src/verdict/llm        OpenAI-compatible client, bounded tool-calling episode
src/verdict/court      roles (claimant, investigator, adversary, auditor) + the loop
src/verdict/domains    plug-ins: claim vocabulary + tools; supply_chain is the first
evals/                 historical incidents with known answers
```

## License

Apache-2.0
