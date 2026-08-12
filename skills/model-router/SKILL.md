# Model Router Skill

Automatically classifies an incoming task and selects the best free-tier model
from available providers (Groq → Gemini → Nous → OpenRouter free pool).

## Usage

Call this skill as the FIRST action on any incoming task. It returns a routing
decision including the recommended model, provider, switch command, and reasoning.

## How it works

1. **Classify** — reads the task text, matches against task-type keywords
2. **Score** — ranks all available (provider, model) candidates by fit + cost
3. **Pick** — selects the highest-scoring free-tier model
4. **Output** — returns: task_type, provider, model, switch command, confidence, reasoning

## Files

- `routing-catalog.json` — provider/model catalog (free-first order)
- `routing-rules.json` — task type definitions + scoring rules
- `routing-state.json` — live state (rate limits, history, session counters)
- `routing-router.py` — the router core (classify → score → pick)

## Invocation

The agent should call this skill before processing any incoming task:

```
# Run the router on the task text
python3 routing-router.py "<task description>" --json

# Or with an explicit task type tag if the user specified one
python3 routing-router.py "<task description>" --tag code --json
```

## Output format

The router emits JSON with these fields:

| Field | Description |
|-------|-------------|
| `task_type` | Classified type: simple, code, reasoning, long_context, creative, summary |
| `chosen_provider` | Best provider: groq, gemini, nous, openrouter |
| `chosen_model` | Raw model id (e.g. llama-3.3-70b-versatile) |
| `chosen_alias` | Friendly tier alias (e.g. large, flash, small) |
| `switch_command` | The `/model` command to switch this session |
| `confidence` | Classification confidence score |
| `ask_user` | Whether the router is confused and wants user input |
| `ask_prompt` | Question to ask the user if confused |
| `reasoning` | List of strings explaining the decision |
| `all_candidates` | All scored candidates for transparency |

## Switch command format

The router produces one of these switch commands:

- `/model groq/llama-3.3-70b-versatile` — direct provider/model form
- `/model gemini/gemini-3.6-flash` — direct provider/model form
- `/model nous/upstage/solar-pro4:free` — direct provider/model form
- `/model openrouter/nvidia/nemotron-3-ultra-550b-a55b:free` — direct provider/model form

These are in the `provider/model_id` short form that Hermes accepts for
session-scoped model switching.

## Confusion handling

When the router can't decide confidently (score below threshold, close runner-up),
it sets `ask_user: true` and provides an `ask_prompt`. The agent should ask the
user before proceeding. If the user specifies a type, re-run with `--tag <type>`.

## Mid-session switching

`/model <command>` works mid-session on CLI and Telegram — subsequent turns use
the new model. The current turn's response still comes from the pre-switch model,
but all future turns use the routed model.

## Task type reference

| Type | Best free provider | Typical model | Use when |
|------|-------------------|---------------|----------|
| simple | groq | llama-3.1-8b-instant | greetings, short Q&A, confirmations |
| code | groq | llama-3.3-70b-versatile | coding, debugging, scripts, APIs |
| reasoning | gemini | gemini-3.6-flash | analysis, planning, research, math |
| long_context | gemini | gemini-3.6-flash | large documents, big context |
| creative | groq | llama-3.3-70b-versatile | writing, brainstorming, poetry |
| summary | groq | llama-3.1-8b-instant | summarization, extraction, tl;dr |

## Provider fallback chain

If a provider is rate-limited or unavailable, the router falls back in this order:
`groq → gemini → nous → openrouter` (all free tier).

## State tracking

The router records each decision in `routing-state.json`:
- Per-provider rate limit hit counters + cooldowns
- Session task count + model switch count
- Task history (last 50)

When a provider hits 3 rate-limit hits, it's cooldowned for 5 minutes.
