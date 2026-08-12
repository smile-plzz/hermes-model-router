# Hermes Model Router

Automatic free-tier model routing for Hermes Agent — classifies incoming tasks
by context and picks the best available free model across Groq, Gemini, Mistral,
Nous, and OpenRouter.

![Routing Demo](docs/demo.gif)

## What It Does

Every time a task comes in, the router:

1. **Classifies** the task type (simple, code, reasoning, creative, summary, long_context)
2. **Scores** all available (provider, model) candidates by fit + free-tier preference
3. **Picks** the highest-scoring free model
4. **Outputs** the routing decision + `/model` switch command

The goal: zero cost. Every model in the catalog is on a free tier.

## Provider Pool

| Provider | Free Models | Best For |
|----------|------------|----------|
| **Groq** | llama-3.1-8b-instant, llama-3.3-70b-versatile, gpt-oss-120b/20b, allam-2-7b, compound-mini | Code, general chat, creative (fast) |
| **Gemini** | gemini-3.6-flash, gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-flash-latest, gemma-4-26b-a4b-it | Reasoning, long context, creative |
| **Mistral** | mistral-large/medium/small-latest, codestral-latest, devstral-latest, ministral-3b/8b/14b-latest | Code, reasoning, general (strong all-rounder) |
| **Nous** | solar-pro4:free (default) | General fallback |
| **OpenRouter** | nvidia/nemotron-3-ultra-550b:free, gemma-4-26b-a4b-it:free, nemotron-3-nano-30b:free, nemotron-3.5-lightning:free | Ultra-large model when needed |

**Routing priority (free-first):** Groq → Gemini → Mistral → Nous → OpenRouter

## Quick Start

### 1. Install

```bash
# Clone
git clone https://github.com/smile-plzz/hermes-model-router.git
cd hermes-model-router

# Copy files to your Hermes home
cp routing-catalog.json routing-rules.json routing-state.json routing-router.py $HERMES_HOME/
mkdir -p $HERMES_HOME/skills/model-router
cp skills/model-router/SKILL.md $HERMES_HOME/skills/model-router/
```

### 2. Configure API Keys

Add your free-tier API keys to `$HERMES_HOME/.env`:

```bash
# Groq (fast, free tier)
GROQ_API_KEY=gsk_...

# Google AI Studio (Gemini, free tier)
GOOGLE_API_KEY=AQ....
GEMINI_API_KEY=AQ....

# Mistral AI (free tier)
MISTRAL_API_KEY=fTgK...

# OpenRouter (aggregate, free models available)
OPENROUTER_API_KEY=sk-or-v1-...

# Nous (already configured via OAuth)
```

### 3. Register Hermes Aliases (optional but recommended)

```bash
# These let you switch models with short names like /model groq-code
hermes config set model.aliases.groq-code openrouter/groq/llama-3.3-70b-versatile
hermes config set model.aliases.groq-small openrouter/groq/llama-3.1-8b-instant
hermes config set model.aliases.gemini-flash openrouter/gemini/gemini-3.6-flash
hermes config set model.aliases.mistral-large openrouter/mistral/mistral-large-latest
```

### 4. Test the Router

```bash
cd $HERMES_HOME
python3 routing-router.py "write a python function that merges two sorted lists"
```

```
📋 TASK: write a python function that merges two sorted lists
🏷️  TYPE: code
🎯 PROVIDER: groq
🤖 MODEL: llama-3.3-70b-versatile
🔄 SWITCH: /model groq/llama-3.3-70b-versatile
📊 CONFIDENCE: 50
------------------------------------------------------------
REASONING:
  • Classified as 'code': Code generation, review, debugging, scripts
  • Classification score: 50 (threshold to ask: 12)
  • Chosen: groq → llama-3.3-70b-versatile (score: 65.0)
------------------------------------------------------------
CANDIDATES considered:
  - groq         | llama-3.3-70b-versatile             | tier=large    | score=65.0
  - mistral      | mistral-large-latest                | tier=large    | score=59.5
  - gemini       | gemini-3.6-flash                    | tier=flash    | score=57.0
  - nous         | upstage/solar-pro4:free             | tier=pro      | score=56.0
  - openrouter   | nvidia/nemotron-3-ultra-550b-a55b:free | tier=ultra    | score=55.5
============================================================
```

### 5. Use in Hermes Sessions

The skill is auto-loaded by Hermes. Before processing any task, Alfred (or any
agent) calls the router:

```
# Agent calls this as first action on each turn:
python3 routing-router.py "<task description>" --json

# Then switches model:
/model groq/llama-3.3-70b-versatile
```

Or use the `--json` flag for programmatic integration:

```bash
python3 routing-router.py "your task" --json | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d['switch_command'])  # /model groq/llama-3.3-70b-versatile
"
```

## Task Types

| Type | Keywords | Default Provider | Example |
|------|----------|------------------|---------|
| `simple` | hello, hi, thanks, quick, confirm, done | Groq (8b) | "hey what's up", "thanks" |
| `code` | code, function, bug, debug, refactor, python, API | Groq (70b) or Mistral (large) | "write a function to merge sorted lists" |
| `reasoning` | analyze, plan, strategy, pros and cons, evaluate | Gemini (flash) or Mistral (large) | "microservices vs monolith pros and cons" |
| `creative` | story, poem, brainstorm, ideas, write, creative | Groq (70b) or Mistral (large) | "write a poem about rain in Dhaka" |
| `summary` | summarize, tl;dr, extract, key points, condense | Groq (8b) | "summarize this in one sentence" |
| `long_context` | long document, entire, big context, many pages | Gemini (flash, 1M context) | "summarize this 200-page document" |

## Configuration

### routing-catalog.json

Provider and model catalog. Edit to add/remove providers or models.

```json
{
  "providers": {
    "groq": { "priority": 1, "free": true, "models": [...] },
    "gemini": { "priority": 2, "free": true, "models": [...] },
    "mistral": { "priority": 3, "free": true, "models": [...] },
    "nous": { "priority": 4, "free": true, "models": [...] },
    "openrouter": { "priority": 5, "free": true, "models": [...] }
  },
  "provider_order_free_first": ["groq", "gemini", "mistral", "nous", "openrouter"]
}
```

### routing-rules.json

Task type definitions, keywords, and scoring rules.

```json
{
  "task_types": {
    "code": {
      "keywords": ["code", "function", "bug", "debug", ...],
      "default_tier": "large",
      "prefer_provider": "groq"
    }
  },
  "scoring": {
    "keyword_match_bonus": 10,
    "free_provider_bonus": 8,
    "rate_limit_penalty": -20
  }
}
```

### routing-state.json

Live state — rate limit counters, session stats, task history.

```json
{
  "providers": {
    "groq": { "rate_limit_hits": 0, "last_used": "...", "cooldown_until": null },
    "gemini": { "rate_limit_hits": 0, "last_used": "...", "cooldown_until": null }
  },
  "session": { "tasks": 42, "model_switches": 12, "cost_estimate": 0.0 },
  "history": [ ... ]
}
```

## Running the Router

```bash
# Basic usage
python3 routing-router.py "your task description"

# JSON output (for programmatic use)
python3 routing-router.py "your task" --json

# Explicit task type tag
python3 routing-router.py "your task" --tag code

# Ask when confused (interactive)
python3 routing-router.py "hey write a plan" --ask-if-confused

# List all available providers and models
python3 routing-router.py --list-providers

# Read task from stdin
echo "write a function" | python3 routing-router.py
```

## Mid-Session Model Switching

`/model <name>` works mid-session on CLI and Telegram. The current turn's
response still comes from the pre-switch model, but all subsequent turns use
the routed model. This means:

- Message N: routed by the router, responded to by the previous model
- Message N+1: uses the correctly routed model

For a flowing conversation, this is close enough — the classification and
decision are automatic, and the model is correct within one turn.

## How Classification Works

The router matches task text against keyword lists for each task type:

- **Multi-word keywords** (phrases like "pros and cons"): substring match
- **Single-word keywords ≥4 chars** (like "debug"): substring match catches variants ("debugging")
- **Short keywords <4 chars** (like "go"): word-boundary match only (avoids false positives)

Scoring:
- Each keyword match: +10 points (phrase) or +3 (short word)
- Free provider bonus: +8
- Preferred provider for task type: +5
- Rate limit penalty: -20 per hit
- Below confidence threshold (12): asks user if genuinely ambiguous

## Rate Limit & Cooldown

When a provider returns rate limit errors:
- Each hit increments `rate_limit_hits`
- At 3 hits: provider enters 5-minute cooldown
- Cooldown providers are skipped during routing
- Cooldown expires automatically after duration

## Files

```
hermes-model-router/
├── routing-catalog.json       # Provider + model catalog
├── routing-rules.json          # Task types + scoring rules
├── routing-state.json          # Live state (rate limits, history)
├── routing-router.py           # Router core (classify → score → pick)
├── skills/
│   └── model-router/
│       └── SKILL.md            # Hermes skill wrapper
├── examples/
│   └── demo.sh                 # Quick demo script
└── README.md
```

## Demo

```bash
# Run the demo script
bash examples/demo.sh
```

This runs 10 sample tasks through the router and prints a table of decisions.

## Integrating as an Automatic Gateway Hook

For true "always active" automatic routing on Telegram/Discord, wire the
router into the Hermes gateway as a pre-task hook. The router script is the
core — the hook is the plumbing:

```python
# Pseudo-code for gateway integration
def on_incoming_message(message):
    decision = run_router(message.text)
    if decision["ask_user"]:
        ask_user(decision["ask_prompt"])
    else:
        switch_model(decision["switch_command"])
    # Now process the message on the routed model
```

The current prototype is a script + skill that agents call before each task.
Gateway hook integration is the next step.

## License

MIT
