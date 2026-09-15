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

Early. Being built in the open, one commit at a time. See `docs/architecture.md` for the design
and `docs/roadmap.md` for what's next.

## Quick start

```bash
uv sync --all-extras
cp .env.example .env         # point it at Ollama, Groq, OpenRouter or any OpenAI-compatible API
uv run verdict ask "Is requests 2.28.0 safe to adopt?" --domain supply-chain
```

## License

Apache-2.0
