# Hermes Model Router

Automatic free-tier model routing for Hermes Agent. Classifies an incoming task,
picks the best free model for it across Groq, Gemini, Mistral, Nous and
OpenRouter, and either prints the `/model` switch command or calls the model
directly and returns the answer.

The goal is zero cost: every model in the catalog is on a free tier.

## How it works

```
task text
   │
   ├─ classify ──→ task type (simple / code / reasoning / creative / summary / long_context)
   │                each type declares a target capability class + wanted strengths
   │
   ├─ filter ────→ drop providers with no API key, and providers in cooldown
   │
   ├─ score ─────→ every remaining model, on class fit + strengths + context +
   │                provider preference + free-tier + recent rate limits
   │
   └─ pick ──────→ highest score, ties broken by catalog order
```

Models are described on a **shared capability scale**, not per-provider tier
names, so candidates from different providers are actually comparable:

- `class`: `tiny` → `small` → `medium` → `large` → `xlarge`
- `strengths`: `fast`, `code`, `reasoning`, `creative`, `long_context`
- `context`: usable context window

A task type asks for a target class and the strengths it cares about. Being one
class *under* target is penalised harder than being one class over, so the
router degrades toward a bigger model rather than a weaker one.

## Quick start

```bash
git clone https://github.com/smile-plzz/hermes-model-router.git
cd hermes-model-router
cp .env.example .env      # fill in whichever keys you have

python router.py --doctor            # check config, keys and routing health
python router.py "write a python function that merges two sorted lists"
python respond.py "explain monads in two sentences"
```

Nothing needs to be copied anywhere to try it — the router finds its config in
its own directory. See [Deploying into Hermes](#deploying-into-hermes) for the
always-on setup.

## Commands

### `router.py` — decide only

```bash
python router.py "your task"                # human-readable decision
python router.py "your task" --json         # machine-readable
python router.py "your task" --tag code     # skip classification
python router.py --list                     # every provider and model
python router.py --doctor                   # health check
python router.py --doctor --live            # also verify model ids against the provider APIs
```

Useful flags: `--config-dir` (where the `routing-*.json` live), `--state-path`,
`--no-record` (don't touch the state file), `--ignore-keys` (score providers
even without a key configured).

Example:

```
==================================================================
TASK       : write a python function that merges two sorted lists
TYPE       : code  (confidence 78)
PROVIDER   : groq
MODEL      : llama-3.3-70b-versatile
SWITCH     : /model groq/llama-3.3-70b-versatile
------------------------------------------------------------------
REASONING:
  - Classified 'code' (score 20, margin 20, confidence 78) via: function, python
  - Chose groq/llama-3.3-70b-versatile (class large, score 67.5)
------------------------------------------------------------------
TOP CANDIDATES:
     67.5  groq        llama-3.3-70b-versatile      large  (strengths code; preferred provider)
     60.0  gemini      gemini-3.6-flash             large  (strengths code)
     58.5  groq        openai/gpt-oss-120b          xlarge (class xlarge vs target large; ...)
==================================================================
```

### `respond.py` — decide *and* answer

Routes, calls the winning model, and falls back down the ranked list when a
provider fails. What happened is written back to the state file, so a provider
that rate-limits you actually cools down.

```bash
python respond.py "your task"           # answer + which model produced it
python respond.py "your task" --raw     # answer only, for piping
python respond.py "your task" --json    # full result incl. every attempt
```

Failover is one model per provider (default 4 attempts, `--max-attempts`) —
retrying a smaller model behind the same rate limit is not worth the latency.

### `--doctor`

The health check, and the first thing to run when routing misbehaves:

- validates the catalog and rules against each other (unknown classes, unknown
  strengths, task types pointing at providers that don't exist)
- reports which providers have a usable key and which are in cooldown
- routes six probe tasks so you can see the classifier's actual behaviour
- with `--live`, calls each provider's model-list endpoint and reports catalog
  entries that no longer exist

## Task types

| Type | Target class | Wants | Prefers | Example |
|------|-------------|-------|---------|---------|
| `simple` | small | fast | Groq | "hey what's up", "thanks" |
| `summary` | small | fast | Groq | "summarize this in one sentence" |
| `code` | large | code | Groq | "write a function to merge sorted lists" |
| `reasoning` | large | reasoning | Gemini | "microservices vs monolith, pros and cons" |
| `creative` | large | creative | Groq | "write a poem about rain in Dhaka" |
| `long_context` | large | long_context (≥500k ctx) | Gemini | "summarize this 200-page document" |

`simple` only applies to messages of 8 words or fewer, so a greeting in front of
real work ("hey, now refactor the auth module…") cannot downgrade the model.

## Classification

Each task type declares `strong` keywords (10 points) and `weak` ones (3).
Matching rules:

- **phrases** ("pros and cons") — substring match
- **words ≥4 chars** ("debug") — prefix match on a word boundary, so it catches
  "debugging" but "large" does not fire inside "enlarge"
- **words <4 chars** ("yo") — exact token match only

The highest total wins. Ties are broken by each type's declared `precedence`,
never by dictionary order — `code` outranks `simple`, so "write a one-line
python hello world" is code, not chat.

`confidence` (0–100) combines the winner's share of all matched evidence with
how much evidence there was. When the margin over the runner-up is small *and*
confidence is low, the decision carries `ask_user: true` and a prompt listing
the plausible types. The decision is still usable — asking is a suggestion, not
a blocker.

## Configuration

| File | Purpose |
|------|---------|
| `routing-catalog.json` | providers, endpoints, key names, models with class/strengths/context |
| `routing-rules.json` | task types, keywords, scoring weights, ambiguity thresholds |
| `routing-state.json` | live state: rate-limit counters, cooldowns, session stats, history |

Config is looked up in this order, first hit wins:

1. `--config-dir` (if given, it must contain the files — a typo is an error, not
   a silent fallback)
2. `$HERMES_ROUTER_HOME`
3. `$HERMES_HOME`
4. the directory holding `router.py`

API keys come from the real environment first, then from a `.env` in any
directory on that same search path.

Both config files carry a `version`. If it doesn't match what the code expects
the router refuses to start and tells you where the stale copy is, rather than
failing later with a confusing error.

### Adding a model

Add it to the right provider in `routing-catalog.json`:

```json
{ "id": "some-new-model", "class": "large", "context": 128000,
  "strengths": ["code", "reasoning"] }
```

Then `python router.py --doctor --live` to confirm the id is real and the config
still validates. Order within a provider is the tie-break, so put the ones you
prefer first.

## Rate limits and cooldown

- a `429`, or a `400` that mentions quota, increments the provider's counter
- 3 hits puts the provider in a 5-minute cooldown; further hits double it
- providers in cooldown are removed from scoring entirely
- any success resets the counter and clears the cooldown

`auth` failures are reported but never trigger a cooldown — a bad key is not a
rate limit, and retrying won't fix it.

## Deploying into Hermes

```bash
cp routing-catalog.json routing-rules.json router.py providers.py respond.py "$HERMES_HOME/"
mkdir -p "$HERMES_HOME/skills/model-router"
cp skills/model-router/SKILL.md "$HERMES_HOME/skills/model-router/"
```

The agent then calls the router as its first action on each turn:

```bash
python router.py "<task text>" --json    # then apply .switch_command
```

Note on mid-session switching: `/model` takes effect from the *next* turn. The
turn that triggered the switch is still answered by the previous model. Use
`respond.py` instead when you need the routed model to answer immediately —
it calls the chosen model directly and sidesteps the delay.

## Using it as a library

```python
from router import Router

router = Router()
decision = router.route("write a python function to merge two sorted lists")
print(decision["provider"], decision["model"], decision["switch_command"])

from respond import respond
result = respond(router, "explain monads in two sentences")
print(result["response"])
router.save_state()
```

## Tests

```bash
python -m pytest tests -q
```

47 tests, no network access — provider calls are stubbed. They cover
classification (including the regressions this router was built to fix),
candidate scoring, cooldown handling, key filtering, config resolution, HTTP
error classification and end-to-end failover.

## Layout

```
hermes-model-router/
├── router.py                   # config, classification, scoring, state, CLI
├── providers.py                # HTTP adapters + error classification
├── respond.py                  # route → call → failover → record
├── routing-catalog.json        # providers and models
├── routing-rules.json          # task types and scoring
├── routing-state.json          # live state (gitignored)
├── skills/model-router/SKILL.md
├── examples/demo.py
└── tests/test_router.py
```

## License

MIT
