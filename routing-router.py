#!/usr/bin/env python3
"""
Hermes model routing prototype.

Reads task description (and optional user tags), classifies it,
checks provider state, scores candidates, picks the best free
model, and outputs a routing decision.

Usage:
  python routing-router.py "write a python script to parse csv"
  python routing-router.py "summarize this long article" --json
  python routing-router.py "hello" --json
  python routing-router.py "analyze the architecture" --ask-if-confused

Outputs a recommended model alias + provider + task type.
When --json is passed, emits a machine-readable JSON decision.
When the router is genuinely confused, it prints an ask-user prompt
and returns a safe default (unless --require-confirmation is set).
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths — resolve relative to HERMES_HOME so this works in any session
# ---------------------------------------------------------------------------
HERMES_HOME = os.environ.get("HERMES_HOME") or str(
    Path.home() / "AppData" / "Local" / "hermes"
)
CATALOG_PATH = Path(HERMES_HOME) / "routing-catalog.json"
RULES_PATH = Path(HERMES_HOME) / "routing-rules.json"
STATE_PATH = Path(HERMES_HOME) / "routing-state.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_task(
    text: str, rules: dict, task_types: dict[str, Any]
) -> tuple[str | None, dict[str, int] | None, int]:
    """
    Return (best_type, per_type_scores, top_score).
    best_type is None if no type matched at all.

    Matching rules:
      - Multi-word keywords: substring match in text (phrase detection).
      - Single-word keywords (len >= 4): substring match in text — catches
        compound forms like "debugging" containing "debug".
      - Short single-word keywords (len < 4): word-boundary match only —
        avoids false positives like "c" inside "document" or "go" inside
        "going". Checked against the token set extracted from the text.
    """
    text_lower = text.lower()
    words = set(re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text_lower))

    type_scores: dict[str, int] = {}

    for type_name, type_def in task_types.items():
        score = 0
        keywords = type_def.get("keywords", [])
        for kw in keywords:
            kw_lower = kw.lower()
            is_phrase = " " in kw_lower  # multi-word keyword

            if is_phrase:
                # Phrase: substring match in the full text
                if kw_lower in text_lower:
                    score += rules["scoring"]["keyword_match_bonus"]
            else:
                # Single word
                if len(kw_lower) >= 4:
                    # Longer word: substring match catches variants
                    if kw_lower in text_lower:
                        score += rules["scoring"]["keyword_match_bonus"]
                else:
                    # Short word: word-boundary match against tokens only
                    if kw_lower in words:
                        score += rules["scoring"]["keyword_partial_bonus"]

        if score > 0:
            type_scores[type_name] = score

    if not type_scores:
        return None, None, 0

    best_type = max(type_scores, key=type_scores.get)
    top_score = type_scores[best_type]
    return best_type, type_scores, top_score


# ---------------------------------------------------------------------------
# Scoring & selection
# ---------------------------------------------------------------------------
def provider_available(provider: str, state: dict) -> bool:
    """Check cooldown / rate-limit state for a provider."""
    prov_state = state.get("providers", {}).get(provider, {})
    cooldown_until = prov_state.get("cooldown_until")
    if cooldown_until and datetime.fromisoformat(cooldown_until) > datetime.now(timezone.utc):
        return False
    return True


def score_candidate(
    provider: str,
    model_tier: str,
    catalog: dict,
    rules: dict,
    state: dict,
    task_type: str | None,
    score: int,
) -> float:
    """
    Score a (provider, model) candidate.
    Higher = better.
    """
    sc = rules["scoring"]
    cand_score = 0.0

    # Base score from classification confidence
    cand_score += score * 1.0

    # Free provider bonus (we want free of cost)
    prov_info = catalog.get("providers", {}).get(provider, {})
    if prov_info.get("free", False):
        cand_score += sc["free_provider_bonus"]

    # Prefer-provider bonus if this provider is preferred for the task type
    if task_type:
        type_def = rules.get("task_types", {}).get(task_type, {})
        if type_def.get("prefer_provider") == provider:
            cand_score += sc["prefer_provider_bonus"]

    # Rate-limit penalty
    prov_state = state.get("providers", {}).get(provider, {})
    rate_hits = prov_state.get("rate_limit_hits", 0)
    if rate_hits > 0:
        cand_score += sc["rate_limit_penalty"] * rate_hits

    # Priority order bonus (lower priority number = earlier in free list)
    order = catalog.get("provider_order_free_first", [])
    if provider in order:
        idx = order.index(provider)
        cand_score += (len(order) - idx) * 0.5

    return cand_score


# ---------------------------------------------------------------------------
# Alias resolution for Hermes /model switching
# ---------------------------------------------------------------------------
def resolve_hermes_alias(provider: str, model_id: str, catalog: dict) -> str | None:
    """Given a provider and raw model id, return the Hermes alias string.
    Format: <provider_slug>/<model_id>  — e.g. "groq/llama-3.3-70b-versatile"
    This is what /model <alias> expects when using the short form.
    """
    prov = catalog.get("providers", {}).get(provider, {})
    alias_for = prov.get("alias_for", {})
    # alias_for maps alias_name -> model_id; invert it
    for alias_name, mid in alias_for.items():
        if mid == model_id:
            # Return short-form alias: provider/model_id
            return f"{provider}/{model_id}"
    # Fallback: just use provider/model_id directly
    return f"{provider}/{model_id}"


def hermes_model_switch_cmd(provider: str, model_id: str, catalog: dict) -> str:
    """Return the CLI command to switch Hermes to this model."""
    alias = resolve_hermes_alias(provider, model_id, catalog)
    return f"/model {alias}"


def pick_model(
    text: str,
    catalog: dict,
    rules: dict,
    state: dict,
    user_tag: str | None = None,
) -> dict:
    """
    Main routing logic.

    Returns a decision dict with:
      - task_type
      - chosen_provider
      - chosen_model (raw provider model id)
      - chosen_alias (friendly alias, or None)
      - reasoning (list of strings)
      - confidence
      - ask_user (bool) — whether the router wants to ask the user
      - ask_prompt (str | None)
      - fallback_used (bool)
    """
    task_types = rules.get("task_types", {})
    scoring = rules.get("scoring", {})
    providers = catalog.get("providers", {})
    provider_order = catalog.get("provider_order_free_first", [])
    fallbacks = catalog.get("fallbacks", {})

    # 1. Classify
    best_type, type_scores, top_score = classify_task(text, rules, task_types)

    reasoning: list[str] = []
    if best_type:
        type_def = task_types[best_type]
        reasoning.append(f"Classified as '{best_type}': {type_def.get('description', '')}")
        reasoning.append(f"Classification score: {top_score} (threshold to ask: {scoring.get('min_confidence_to_skip_ask', 12)})")
    else:
        reasoning.append("No task type matched — using default tier")

    # 2. Decide if we should ask the user
    ask_user = False
    ask_prompt = None
    confidence = top_score

    if best_type and top_score < scoring.get("min_confidence_to_skip_ask", 12):
        # Confused — check if there's a runner-up within threshold
        if len(type_scores) >= 2:
            sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
            runner_up_score = sorted_types[1][1]
            gap = top_score - runner_up_score
            if gap <= scoring.get("ask_user_threshold", 6):
                ask_user = True
                options = [t for t, _ in sorted_types[:3]]
                ask_prompt = scoring.get("when_confused", {}).get("prompt_template", "This task could be [{options}]. Which fits best?")
                ask_prompt = ask_prompt.replace("{options}", " / ".join(options))
                reasoning.append(f"AMBIGUOUS: close call between {options} — asking user")
    elif not best_type:
        ask_user = True
        ask_prompt = "I couldn't figure out what kind of task this is. What type is it? (simple / code / reasoning / long_context / creative / summary / other)"
        reasoning.append("AMBIGUOUS: no type matched — asking user")

    # 3. Pick a tier
    if user_tag:
        # User explicitly specified a task type via tag
        if user_tag in task_types:
            best_type = user_tag
            type_def = task_types[user_tag]
            reasoning.append(f"User specified task type: '{user_tag}'")
            confidence = 99
        else:
            reasoning.append(f"User tag '{user_tag}' not recognized — falling back to classification")

    if not best_type:
        best_type = "simple"  # safe default
        type_def = task_types.get("simple", {})
        reasoning.append(f"Using default task type: '{best_type}'")

    tier = type_def.get("default_tier", scoring.get("default_tier_if_ambiguous", "small"))
    prefer_provider = type_def.get("prefer_provider", None)
    fallback_tier = type_def.get("fallback_tier", tier)

    # 4. Score candidates across providers
    candidates: list[dict] = []
    for provider in provider_order:
        if not provider_available(provider, state):
            reasoning.append(f"SKIP {provider}: in cooldown")
            continue
        prov_models = providers.get(provider, {}).get("models", [])
        alias_for = providers.get(provider, {}).get("alias_for", {})

        # Find a model matching the preferred tier
        tier_model = None
        for m in prov_models:
            if m.get("tier") == tier:
                tier_model = m
                break

        # Fallback to any model on this provider if tier model not found
        chosen_model = tier_model
        used_fallback_tier = False
        if not chosen_model:
            # Try fallback tier
            for m in prov_models:
                if m.get("tier") == fallback_tier:
                    chosen_model = m
                    used_fallback_tier = True
                    break
        if not chosen_model and prov_models:
            chosen_model = prov_models[0]
            used_fallback_tier = True

        if not chosen_model:
            continue

        model_id = chosen_model["id"]
        candidate_score = score_candidate(
            provider, tier, catalog, rules, state, best_type, top_score
        )
        if used_fallback_tier:
            candidate_score += scoring.get("fallback_used_penalty", -3)

        candidates.append({
            "provider": provider,
            "model_id": model_id,
            "tier": chosen_model.get("tier", "unknown"),
            "score": candidate_score,
            "fallback_tier_used": used_fallback_tier,
        })

    if not candidates:
        # Absolute fallback — pick first available provider with any model
        for provider in provider_order:
            prov_models = providers.get(provider, {}).get("models", [])
            if prov_models and provider_available(provider, state):
                candidates.append({
                    "provider": provider,
                    "model_id": prov_models[0]["id"],
                    "tier": prov_models[0].get("tier", "unknown"),
                    "score": -1.0,
                    "fallback_tier_used": True,
                })
                reasoning.append(f"ABSOLUTE FALLBACK: {provider} ({prov_models[0]['id']})")
                break

    if not candidates:
        return {
            "task_type": best_type,
            "chosen_provider": None,
            "chosen_model": None,
            "chosen_alias": None,
            "reasoning": reasoning,
            "confidence": confidence,
            "ask_user": True,
            "ask_prompt": "No accessible provider found. Check your API keys.",
            "fallback_used": True,
        }

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    # 5. Build the decision
    chosen_provider = best["provider"]
    chosen_model = best["model_id"]

    # Try to find a friendly alias for this model on this provider
    alias_for = providers.get(chosen_provider, {}).get("alias_for", {})
    chosen_alias = None
    for alias, model_id in alias_for.items():
        if model_id == chosen_model:
            chosen_alias = alias
            break

    reasoning.append(f"Chosen: {chosen_provider} → {chosen_model} (score: {best['score']:.1f})")
    if best.get("fallback_tier_used"):
        reasoning.append(f"Used fallback tier '{fallback_tier}' — preferred tier '{tier}' not available on {chosen_provider}")

    decision = {
        "task_type": best_type,
        "chosen_provider": chosen_provider,
        "chosen_model": chosen_model,
        "chosen_alias": chosen_alias,
        "reasoning": reasoning,
        "confidence": confidence,
        "ask_user": ask_user,
        "ask_prompt": ask_prompt,
        "fallback_used": best.get("fallback_tier_used", False),
        "all_candidates": [
            {"provider": c["provider"], "model": c["model_id"], "tier": c["tier"], "score": round(c["score"], 2)}
            for c in candidates
        ],
    }
    return decision


# ---------------------------------------------------------------------------
# State update helpers
# ---------------------------------------------------------------------------
def record_task(state: dict, decision: dict, text: str, user_tag: str | None = None) -> dict:
    """Record a task in history and update provider state."""
    providers = state.get("providers", {})
    prov = decision.get("chosen_provider")
    if prov and prov in providers:
        providers[prov]["last_used"] = now_iso()

    session = state.get("session", {})
    session["tasks"] = session.get("tasks", 0) + 1
    if decision.get("chosen_alias") != decision.get("previous_alias"):
        session["model_switches"] = session.get("model_switches", 0) + 1

    history = state.get("history", [])
    entry = {
        "task": text[:200],
        "chosen_provider": prov,
        "chosen_model": decision.get("chosen_model"),
        "chosen_alias": decision.get("chosen_alias"),
        "task_type": decision.get("task_type"),
        "confidence": decision.get("confidence"),
        "timestamp": now_iso(),
        "user_tag": user_tag,
    }
    history.append(entry)
    max_hist = state.get("config", {}).get("max_history", 50)
    if len(history) > max_hist:
        history = history[-max_hist:]

    state["providers"] = providers
    state["session"] = session
    state["history"] = history
    return state


def record_rate_limit(state: dict, provider: str) -> dict:
    """Increment rate-limit counter for a provider and set cooldown if threshold hit."""
    providers = state.get("providers", {})
    if provider not in providers:
        return state
    prov_state = providers[provider]
    prov_state["rate_limit_hits"] = prov_state.get("rate_limit_hits", 0) + 1
    threshold = state.get("config", {}).get("cooldown_after_rate_limits", 3)
    if prov_state["rate_limit_hits"] >= threshold:
        cooldown_duration = state.get("config", {}).get("cooldown_duration_seconds", 300)
        from datetime import datetime, timezone, timedelta
        prov_state["cooldown_until"] = (datetime.now(timezone.utc) + timedelta(seconds=cooldown_duration)).isoformat()
    providers[provider] = prov_state
    state["providers"] = providers
    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes model router — pick the best free model for a task")
    parser.add_argument("task", nargs="?", help="Task description text")
    parser.add_argument("--json", action="store_true", help="Emit JSON decision instead of human-readable")
    parser.add_argument("--tag", "-t", help="Explicit task type tag (simple/code/reasoning/long_context/creative/summary)")
    parser.add_argument("--ask-if-confused", action="store_true", help="Print ask prompt and wait for user input when confused")
    parser.add_argument("--record", action="store_true", help="Record the task in state history (default: True in session use)")
    parser.add_argument("--dry-run", action="store_true", help="Don't update state file")
    parser.add_argument("--state-path", help="Override state file path")
    parser.add_argument("--list-providers", action="store_true", help="List available providers and models and exit")

    args = parser.parse_args()

    # Load files
    try:
        catalog = load_json(CATALOG_PATH)
        rules = load_json(RULES_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    state_path = Path(args.state_path) if args.state_path else STATE_PATH
    if state_path.exists():
        state = load_json(state_path)
    else:
        state = {"version": 1, "providers": {}, "session": {}, "history": [], "config": {}}

    if args.list_providers:
        print("=== Available Providers & Models (free-first order) ===")
        for provider in catalog.get("provider_order_free_first", []):
            prov = catalog.get("providers", {}).get(provider, {})
            free_str = "FREE" if prov.get("free") else "PAID"
            print(f"\n[{provider}] ({free_str}, priority {prov.get('priority', '?')})")
            alias_for = prov.get("alias_for", {})
            for model in prov.get("models", []):
                alias = [a for a, m in alias_for.items() if m == model["id"]]
                alias_str = f" alias={alias[0]}" if alias else ""
                print(f"  - {model['id']}  tier={model['tier']}  context={model['context']}  speed={model['speed']}{alias_str}")
        return

    # If no task text, read from stdin
    task_text = args.task
    if not task_text:
        if not sys.stdin.isatty():
            task_text = sys.stdin.read().strip()
        else:
            print("Usage: python routing-router.py \"your task description\"", file=sys.stderr)
            print("       echo \"your task\" | python routing-router.py", file=sys.stderr)
            sys.exit(1)

    # Run routing
    decision = pick_model(task_text, catalog, rules, state, user_tag=args.tag)

    # Handle ask-if-confused interactively
    if decision.get("ask_user") and args.ask_if_confused:
        print(f"\n❓ {decision['ask_prompt']}", file=sys.stderr)
        try:
            response = input("Your answer: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = ""
        if response:
            # Try to map response to a known task type
            response_map = {
                "simple": "simple", "code": "code", "reasoning": "reasoning",
                "long": "long_context", "long_context": "long_context",
                "creative": "creative", "summary": "summary",
                "other": None,
            }
            mapped = response_map.get(response, response)
            if mapped and mapped in rules.get("task_types", {}):
                # Re-run with explicit tag
                decision = pick_model(task_text, catalog, rules, state, user_tag=mapped)
                decision["user_overridden"] = True
                decision["user_response"] = response
            elif response == "other":
                decision["user_overridden"] = True
                decision["user_response"] = "other"
                decision["manual_task_type"] = True

    # Record in state unless dry-run
    if not args.dry_run:
        state = record_task(state, decision, task_text, user_tag=args.tag)
        save_json(state_path, state)

    # Output
    if args.json:
        # Add the switch command for Hermes
        decision["switch_command"] = f"/model {decision['chosen_provider']}/{decision['chosen_model']}"
        print(json.dumps(decision, indent=2, ensure_ascii=False))
    else:
        print("=" * 60)
        print(f"📋 TASK: {task_text[:120]}{'...' if len(task_text) > 120 else ''}")
        print(f"🏷️  TYPE: {decision['task_type']}")
        print(f"🎯 PROVIDER: {decision['chosen_provider']}")
        print(f"🤖 MODEL: {decision['chosen_model']}")
        if decision.get("chosen_alias"):
            print(f"🏷️  ALIAS: {decision['chosen_alias']}")
        print(f"🔄 SWITCH: /model {decision['chosen_provider']}/{decision['chosen_model']}")
        print(f"📊 CONFIDENCE: {decision['confidence']}")
        if decision.get("ask_user"):
            print(f"❓ ASK USER: {decision['ask_prompt']}")
        if decision.get("fallback_used"):
            print("⚠️  FALLBACK USED (preferred tier not available on this provider)")
        print("-" * 60)
        print("REASONING:")
        for r in decision.get("reasoning", []):
            print(f"  • {r}")
        print("-" * 60)
        print("CANDIDATES considered:")
        for c in decision.get("all_candidates", []):
            print(f"  - {c['provider']:12s} | {c['model']:35s} | tier={c['tier']:8s} | score={c['score']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
