---
name: model-router
description: Pick the best free-tier model for an incoming task. Use as the first action on each turn, before doing the work, so the task runs on an appropriately sized free model instead of whatever is currently loaded.
---

# Model Router

Classifies a task and selects the best free model across Groq, Gemini, Mistral,
Nous and OpenRouter. Providers with no API key, and providers currently in
rate-limit cooldown, are excluded automatically.

## Deciding which model to use

Run this first, with the user's task text:

```bash
python router.py "<task text>" --json
```

Then apply `switch_command` (e.g. `/model groq/llama-3.3-70b-versatile`).

If the user already told you the kind of task, skip classification:

```bash
python router.py "<task text>" --tag code --json
```

Valid tags: `simple`, `code`, `reasoning`, `creative`, `summary`, `long_context`.

## Getting an answer directly

`/model` only takes effect on the *next* turn — the current turn is still
answered by the previous model. When the routed model should answer right now,
call it directly instead:

```bash
python respond.py "<task text>" --raw
```

This routes, calls the chosen model, and falls back to the next provider if that
one is rate-limited or down.

## Decision fields

| Field | Meaning |
|-------|---------|
| `task_type` | simple, code, reasoning, creative, summary, long_context |
| `provider` / `model` | the choice; `null` if nothing is available |
| `switch_command` | the `/model` command to apply |
| `class` | capability class of the chosen model (tiny…xlarge) |
| `confidence` | 0–100 classification confidence |
| `ask_user` / `ask_prompt` | the classification was ambiguous — see below |
| `reasoning` | why this decision, including which providers were skipped |
| `candidates` | every scored model, best first |

## When `ask_user` is true

The task looked like more than one type, or matched nothing at all. The decision
is still usable — it is a suggestion, not a blocker.

- Mid-conversation, where the type is obvious from context: proceed with the
  chosen model.
- Genuinely unclear, or the task is expensive to get wrong: ask the user with
  `ask_prompt`, then re-run with `--tag <their answer>`.

Never ask the user twice about the same task.

## When nothing is available

`provider` is `null` when every provider is either unconfigured or in cooldown.
Carry on with the current model and mention it. Do not retry in a loop.

## Troubleshooting

```bash
python router.py --doctor          # config validity, key status, cooldowns, probe routes
python router.py --doctor --live   # also verify model ids against the provider APIs
python router.py --list            # every provider and model
```

If the router reports a config version mismatch, the copy in `$HERMES_HOME` is
stale — refresh `routing-catalog.json` and `routing-rules.json` from the repo.

## Notes

- Config resolves from `--config-dir`, then `$HERMES_ROUTER_HOME`, then
  `$HERMES_HOME`, then the script's own directory.
- State (rate limits, cooldowns, last 50 decisions) lives in
  `routing-state.json` next to the config.
- Use `python`, not `python3` — `python3` is a broken stub on Windows.
