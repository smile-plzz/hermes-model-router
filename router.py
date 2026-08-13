#!/usr/bin/env python3
"""Hermes model router — classify a task, pick the best free model.

Importable (`from router import Router`) and runnable:

  python router.py "write a python function that merges two sorted lists"
  python router.py "summarize this article" --json
  python router.py "hey what's up" --tag simple
  python router.py --doctor --live
  python router.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
CATALOG_NAME = "routing-catalog.json"
RULES_NAME = "routing-rules.json"
STATE_NAME = "routing-state.json"
CONFIG_VERSION = 3


class ConfigVersionError(RuntimeError):
    """Raised when the config on disk predates this router."""


# --------------------------------------------------------------------------
# Config discovery
# --------------------------------------------------------------------------
def config_search_path(explicit: str | os.PathLike[str] | None = None) -> list[Path]:
    """Directories to search for config, most specific first."""
    candidates = [
        explicit,
        os.environ.get("HERMES_ROUTER_HOME"),
        os.environ.get("HERMES_HOME"),
        SCRIPT_DIR,
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _has_config(directory: Path) -> bool:
    return (directory / CATALOG_NAME).is_file() and (directory / RULES_NAME).is_file()


def resolve_config_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """First directory on the search path that holds both config files.

    An explicitly requested directory is never silently skipped — a typo in
    --config-dir must fail rather than fall through to the bundled defaults.
    """
    if explicit is not None:
        chosen = Path(explicit).expanduser()
        if not _has_config(chosen):
            raise FileNotFoundError(
                f"{chosen} does not contain {CATALOG_NAME} and {RULES_NAME}"
            )
        return chosen

    searched = config_search_path(None)
    for d in searched:
        if _has_config(d):
            return d
    raise FileNotFoundError(
        f"Could not find {CATALOG_NAME} + {RULES_NAME} in any of: "
        + ", ".join(str(d) for d in searched)
    )


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: Any) -> datetime | None:
    """Tolerant ISO parse — always returns an aware datetime or None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# API keys
# --------------------------------------------------------------------------
def load_env_keys(search_dirs: Iterable[Path]) -> dict[str, str]:
    """Real environment wins; .env files along the search path fill the gaps."""
    keys = dict(os.environ)
    for directory in search_dirs:
        env_file = directory / ".env"
        if not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k and not keys.get(k):
                keys[k] = v.strip().strip('"').strip("'")
    return keys


def provider_key(provider_def: dict, keys: dict[str, str]) -> str | None:
    for name in provider_def.get("key_env", []):
        value = keys.get(name)
        if value:
            return value
    return None


def provider_is_configured(provider_def: dict, keys: dict[str, str]) -> bool:
    """A provider with no key_env entries (e.g. the local Hermes CLI) needs no key."""
    if not provider_def.get("key_env"):
        return True
    return provider_key(provider_def, keys) is not None


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z0-9+#]+(?:['-][a-z0-9+#]+)*")


def _keyword_hits(text_lower: str, tokens: set[str], keyword: str) -> bool:
    kw = keyword.lower().strip()
    if not kw:
        return False
    if " " in kw or kw.endswith(" "):
        return kw in text_lower
    if len(kw) >= 4:
        # Prefix match on a word boundary: "debug" catches "debugging",
        # but "large" must not fire on "enlarge".
        return re.search(r"\b" + re.escape(kw) + r"\w*", text_lower) is not None
    return kw in tokens


def classify(text: str, rules: dict) -> dict:
    """Score every task type against the text.

    Returns {"scores": {type: score}, "best": type|None, "runner_up": type|None,
             "margin": int, "confidence": int, "matched": {type: [keywords]}}.
    """
    text_lower = text.lower()
    tokens = set(_WORD_RE.findall(text_lower))
    word_count = len(_WORD_RE.findall(text_lower))
    scoring = rules["scoring"]
    strong_w = scoring["keyword_strong"]
    weak_w = scoring["keyword_weak"]

    scores: dict[str, int] = {}
    matched: dict[str, list[str]] = {}

    for name, spec in rules["task_types"].items():
        max_words = spec.get("max_words")
        if max_words is not None and word_count > max_words:
            continue
        score = 0
        hits: list[str] = []
        for kw in spec.get("strong", []):
            if _keyword_hits(text_lower, tokens, kw):
                score += strong_w
                hits.append(kw)
        for kw in spec.get("weak", []):
            if _keyword_hits(text_lower, tokens, kw):
                score += weak_w
                hits.append(kw)
        if score > 0:
            scores[name] = score
            matched[name] = hits

    if not scores:
        return {"scores": {}, "best": None, "runner_up": None,
                "margin": 0, "confidence": 0, "matched": {}}

    # Deterministic ordering: score desc, then declared precedence desc, then name.
    def sort_key(item: tuple[str, int]) -> tuple[int, int, str]:
        name, score = item
        precedence = rules["task_types"][name].get("precedence", 0)
        return (-score, -precedence, name)

    ranked = sorted(scores.items(), key=sort_key)
    best, top = ranked[0]
    runner_up, second = ranked[1] if len(ranked) > 1 else (None, 0)

    total = sum(scores.values())
    share = top / total
    evidence = min(1.0, top / (2 * strong_w))
    confidence = round(100 * share * evidence)

    return {
        "scores": scores,
        "best": best,
        "runner_up": runner_up,
        "margin": top - second,
        "confidence": confidence,
        "matched": matched,
    }


# --------------------------------------------------------------------------
# Candidate scoring
# --------------------------------------------------------------------------
def score_model(
    provider: str,
    provider_def: dict,
    model: dict,
    type_spec: dict,
    catalog: dict,
    rules: dict,
    provider_state: dict,
) -> tuple[float, list[str]]:
    sc = rules["scoring"]
    classes: list[str] = catalog["classes"]
    notes: list[str] = []
    total = 0.0

    target = type_spec.get("target_class", classes[0])
    try:
        distance = classes.index(model["class"]) - classes.index(target)
    except ValueError:
        distance = 0
    if distance == 0:
        total += sc["class_exact_bonus"]
    else:
        step = sc["class_step_under_penalty"] if distance < 0 else sc["class_step_over_penalty"]
        total += sc["class_exact_bonus"] + step * abs(distance)
        notes.append(f"class {model['class']} vs target {target}")

    want = set(type_spec.get("want_strengths", []))
    have = want & set(model.get("strengths", []))
    if have:
        total += min(len(have) * sc["strength_match_bonus"], sc["strength_match_cap"])
        notes.append("strengths " + ",".join(sorted(have)))

    min_context = type_spec.get("min_context")
    if min_context and model.get("context", 0) < min_context:
        total += sc["context_shortfall_penalty"]
        notes.append(f"context {model.get('context', 0)} < {min_context}")

    if type_spec.get("prefer_provider") == provider:
        total += sc["prefer_provider_bonus"]
        notes.append("preferred provider")

    if provider_def.get("free"):
        total += sc["free_provider_bonus"]

    order: list[str] = catalog["provider_order"]
    if provider in order:
        total += (len(order) - order.index(provider)) * sc["provider_order_bonus_step"]

    hits = provider_state.get("rate_limit_hits", 0)
    if hits:
        total += sc["rate_limit_penalty_per_hit"] * hits
        notes.append(f"{hits} rate-limit hit(s)")

    return total, notes


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
class Router:
    def __init__(
        self,
        config_dir: str | os.PathLike[str] | None = None,
        state_path: str | os.PathLike[str] | None = None,
        keys: dict[str, str] | None = None,
    ) -> None:
        self.search_path = config_search_path(config_dir)
        self.config_dir = resolve_config_dir(config_dir)
        self.catalog = _read_json(self.config_dir / CATALOG_NAME)
        self.rules = _read_json(self.config_dir / RULES_NAME)
        self._check_versions()
        self.state_path = Path(state_path) if state_path else self.config_dir / STATE_NAME
        self.keys = load_env_keys(self.search_path) if keys is None else keys
        self.state = self._load_state()

    def _check_versions(self) -> None:
        stale = [
            f"{name} is v{cfg.get('version', 'unknown')}"
            for name, cfg in ((CATALOG_NAME, self.catalog), (RULES_NAME, self.rules))
            if cfg.get("version") != CONFIG_VERSION
        ]
        if stale:
            raise ConfigVersionError(
                f"{' and '.join(stale)}, but this router needs v{CONFIG_VERSION}. "
                f"Config was loaded from {self.config_dir} — copy the current "
                f"{CATALOG_NAME} and {RULES_NAME} there, or point --config-dir elsewhere."
            )

    # -- state ------------------------------------------------------------
    def _load_state(self) -> dict:
        state: dict = {}
        if self.state_path.is_file():
            try:
                state = _read_json(self.state_path)
            except (json.JSONDecodeError, OSError):
                state = {}
        state.setdefault("version", 3)
        state.setdefault("session", {"tasks": 0, "model_switches": 0, "started_at": now_iso()})
        state.setdefault("history", [])
        state.setdefault(
            "config",
            {"cooldown_after_rate_limits": 3, "cooldown_seconds": 300, "max_history": 50},
        )
        providers = state.setdefault("providers", {})
        for name in self.catalog["providers"]:
            providers.setdefault(name, {"rate_limit_hits": 0, "failures": 0,
                                        "last_used": None, "cooldown_until": None})
        return state

    def save_state(self) -> None:
        _write_json_atomic(self.state_path, self.state)

    def provider_state(self, provider: str) -> dict:
        return self.state["providers"].setdefault(
            provider, {"rate_limit_hits": 0, "failures": 0, "last_used": None, "cooldown_until": None}
        )

    def cooling_down(self, provider: str) -> bool:
        until = _parse_ts(self.provider_state(provider).get("cooldown_until"))
        return bool(until and until > datetime.now(timezone.utc))

    def record_rate_limit(self, provider: str) -> None:
        ps = self.provider_state(provider)
        ps["rate_limit_hits"] = ps.get("rate_limit_hits", 0) + 1
        cfg = self.state["config"]
        if ps["rate_limit_hits"] >= cfg["cooldown_after_rate_limits"]:
            seconds = cfg["cooldown_seconds"] * (2 ** (ps["rate_limit_hits"] - cfg["cooldown_after_rate_limits"]))
            ps["cooldown_until"] = (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()

    def record_failure(self, provider: str) -> None:
        ps = self.provider_state(provider)
        ps["failures"] = ps.get("failures", 0) + 1

    def record_success(self, provider: str) -> None:
        ps = self.provider_state(provider)
        ps["last_used"] = now_iso()
        ps["rate_limit_hits"] = 0
        ps["cooldown_until"] = None

    def record_decision(self, decision: dict, text: str) -> None:
        session = self.state["session"]
        session["tasks"] = session.get("tasks", 0) + 1
        last = session.get("last_model")
        chosen = decision.get("model")
        if chosen and chosen != last:
            session["model_switches"] = session.get("model_switches", 0) + 1
            session["last_model"] = chosen
        history = self.state["history"]
        history.append({
            "task": text[:200],
            "task_type": decision.get("task_type"),
            "provider": decision.get("provider"),
            "model": decision.get("model"),
            "confidence": decision.get("confidence"),
            "timestamp": now_iso(),
        })
        max_history = self.state["config"]["max_history"]
        if len(history) > max_history:
            self.state["history"] = history[-max_history:]

    # -- routing ----------------------------------------------------------
    def candidates(self, task_type: str, require_keys: bool = True) -> list[dict]:
        """Every usable model, best first. One entry per model, not per provider."""
        type_spec = self.rules["task_types"][task_type]
        out: list[dict] = []
        for provider in self.catalog["provider_order"]:
            provider_def = self.catalog["providers"].get(provider)
            if provider_def is None:
                continue
            if require_keys and not provider_is_configured(provider_def, self.keys):
                continue
            if self.cooling_down(provider):
                continue
            ps = self.provider_state(provider)
            for rank, model in enumerate(provider_def["models"]):
                score, notes = score_model(
                    provider, provider_def, model, type_spec, self.catalog, self.rules, ps
                )
                out.append({
                    "provider": provider,
                    "model": model["id"],
                    "class": model["class"],
                    "context": model.get("context"),
                    "score": round(score, 2),
                    "notes": notes,
                    "_order": (self.catalog["provider_order"].index(provider), rank),
                })
        # Ties fall back to catalog order, which is curated, not alphabetical.
        out.sort(key=lambda c: (-c["score"], c["_order"]))
        for c in out:
            del c["_order"]
        return out

    def route(self, text: str, tag: str | None = None, require_keys: bool = True) -> dict:
        task_types = self.rules["task_types"]
        ambiguity = self.rules["ambiguity"]
        reasoning: list[str] = []

        result = classify(text, self.rules)
        task_type = result["best"]
        confidence = result["confidence"]
        ask_user = False
        ask_prompt = None

        if tag:
            if tag in task_types:
                reasoning.append(f"Task type forced by --tag: '{tag}'")
                task_type = tag
                confidence = 100
            else:
                reasoning.append(f"Unknown --tag '{tag}' — ignoring, using classification")

        if task_type is None:
            task_type = ambiguity["default_task_type"]
            reasoning.append(f"No keywords matched — defaulting to '{task_type}'")
            ask_user = True
            ask_prompt = ambiguity["prompt_template"].replace(
                "{options}", " / ".join(task_types)
            )
        elif not tag:
            hits = ", ".join(result["matched"].get(task_type, [])[:6])
            reasoning.append(
                f"Classified '{task_type}' (score {result['scores'][task_type]}, "
                f"margin {result['margin']}, confidence {confidence}) via: {hits}"
            )
            if (result["margin"] < ambiguity["ask_when_margin_below"]
                    and confidence < ambiguity["ask_when_confidence_below"]
                    and result["runner_up"]):
                ask_user = True
                options = [t for t, _ in sorted(
                    result["scores"].items(), key=lambda kv: -kv[1])][:3]
                ask_prompt = ambiguity["prompt_template"].replace("{options}", " / ".join(options))
                reasoning.append(f"Ambiguous: {task_type} vs {result['runner_up']}")

        skipped = [p for p in self.catalog["provider_order"] if self.cooling_down(p)]
        for p in skipped:
            reasoning.append(f"Skipped {p}: in cooldown until {self.provider_state(p)['cooldown_until']}")
        if require_keys:
            for p in self.catalog["provider_order"]:
                if not provider_is_configured(self.catalog["providers"][p], self.keys):
                    reasoning.append(f"Skipped {p}: no API key configured")

        ranked = self.candidates(task_type, require_keys=require_keys)
        if not ranked and require_keys:
            reasoning.append("No configured provider available — retrying without key filter")
            ranked = self.candidates(task_type, require_keys=False)

        if not ranked:
            return {
                "task": text,
                "task_type": task_type,
                "provider": None,
                "model": None,
                "switch_command": None,
                "confidence": confidence,
                "ask_user": True,
                "ask_prompt": "No provider is available. Check your API keys and cooldowns.",
                "reasoning": reasoning,
                "candidates": [],
            }

        best = ranked[0]
        reasoning.append(
            f"Chose {best['provider']}/{best['model']} "
            f"(class {best['class']}, score {best['score']})"
        )
        return {
            "task": text,
            "task_type": task_type,
            "provider": best["provider"],
            "model": best["model"],
            "class": best["class"],
            "switch_command": f"/model {best['provider']}/{best['model']}",
            "confidence": confidence,
            "ask_user": ask_user,
            "ask_prompt": ask_prompt,
            "reasoning": reasoning,
            "candidates": ranked,
        }


# --------------------------------------------------------------------------
# Doctor
# --------------------------------------------------------------------------
def doctor(router: Router, live: bool = False) -> int:
    problems = 0
    print(f"config dir : {router.config_dir}")
    print(f"state file : {router.state_path}")
    print(f"catalog v{router.catalog.get('version')}  rules v{router.rules.get('version')}")

    classes = set(router.catalog["classes"])
    known_strengths = set(router.catalog["strengths"])

    print("\n-- config validation --")
    for provider, pdef in router.catalog["providers"].items():
        for model in pdef["models"]:
            if model["class"] not in classes:
                print(f"  FAIL {provider}/{model['id']}: unknown class '{model['class']}'")
                problems += 1
            unknown = set(model.get("strengths", [])) - known_strengths
            if unknown:
                print(f"  FAIL {provider}/{model['id']}: unknown strengths {sorted(unknown)}")
                problems += 1
    for provider in router.catalog["provider_order"]:
        if provider not in router.catalog["providers"]:
            print(f"  FAIL provider_order lists unknown provider '{provider}'")
            problems += 1
    for name, spec in router.rules["task_types"].items():
        if spec.get("target_class") not in classes:
            print(f"  FAIL task type '{name}': unknown target_class '{spec.get('target_class')}'")
            problems += 1
        pref = spec.get("prefer_provider")
        if pref and pref not in router.catalog["providers"]:
            print(f"  FAIL task type '{name}': prefer_provider '{pref}' not in catalog")
            problems += 1
        min_ctx = spec.get("min_context")
        if min_ctx:
            reachable = any(
                m.get("context", 0) >= min_ctx
                for p in router.catalog["providers"].values()
                for m in p["models"]
            )
            if not reachable:
                print(f"  WARN task type '{name}': no model meets min_context {min_ctx}")
    if not problems:
        print("  ok — catalog and rules are internally consistent")

    print("\n-- providers --")
    usable = 0
    for provider in router.catalog["provider_order"]:
        pdef = router.catalog["providers"][provider]
        if not pdef.get("key_env"):
            status = "no key required"
        elif provider_is_configured(pdef, router.keys):
            status = "key found"
        else:
            status = "NO KEY (" + "/".join(pdef["key_env"]) + ")"
        cooling = " [COOLDOWN]" if router.cooling_down(provider) else ""
        if provider_is_configured(pdef, router.keys) and not cooling:
            usable += 1
        print(f"  {provider:<12} {len(pdef['models'])} models  {status}{cooling}")
    if usable == 0:
        print("  FAIL no usable provider")
        problems += 1

    print("\n-- routing smoke test --")
    for probe in ("hey what's up", "write a python function to merge two sorted lists",
                  "analyze the pros and cons of microservices", "summarize this article",
                  "write a poem about rain", "summarize this 200-page document"):
        d = router.route(probe)
        print(f"  {probe[:45]:<47} -> {d['task_type']:<13} {d['provider']}/{d['model']}")

    if live:
        print("\n-- live model check --")
        try:
            import providers as provider_api
        except ImportError as exc:
            print(f"  SKIP cannot import providers.py: {exc}")
            return 1 if problems else 0
        for provider in router.catalog["provider_order"]:
            pdef = router.catalog["providers"][provider]
            if not pdef.get("models_endpoint"):
                print(f"  {provider}: no models endpoint — skipped")
                continue
            if not provider_is_configured(pdef, router.keys):
                print(f"  {provider}: no key configured — skipped")
                continue
            available, error = provider_api.list_models(
                pdef, provider_key(pdef, router.keys) or ""
            )
            if error:
                print(f"  {provider}: FAIL {error}")
                problems += 1
                continue
            missing = [m["id"] for m in pdef["models"] if m["id"] not in available]
            if missing:
                print(f"  {provider}: {len(missing)} catalog model(s) NOT live: {', '.join(missing)}")
                problems += 1
            else:
                print(f"  {provider}: all {len(pdef['models'])} catalog models live")

    print(f"\n{'FAILED' if problems else 'HEALTHY'} — {problems} problem(s)")
    return 1 if problems else 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def read_task(argv_task: str | None) -> str:
    if argv_task:
        return argv_task
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            return text
    print('Usage: python router.py "your task description"', file=sys.stderr)
    raise SystemExit(2)


def print_human(decision: dict) -> None:
    print("=" * 66)
    task = decision["task"]
    print(f"TASK       : {task[:110]}{'...' if len(task) > 110 else ''}")
    print(f"TYPE       : {decision['task_type']}  (confidence {decision['confidence']})")
    print(f"PROVIDER   : {decision['provider']}")
    print(f"MODEL      : {decision['model']}")
    print(f"SWITCH     : {decision['switch_command']}")
    if decision.get("ask_user"):
        print(f"ASK USER   : {decision['ask_prompt']}")
    print("-" * 66)
    print("REASONING:")
    for line in decision["reasoning"]:
        print(f"  - {line}")
    print("-" * 66)
    print("TOP CANDIDATES:")
    for c in decision["candidates"][:6]:
        note = ("  (" + "; ".join(c["notes"]) + ")") if c["notes"] else ""
        print(f"  {c['score']:>7.1f}  {c['provider']:<11} {c['model']:<40} {c['class']}{note}")
    print("=" * 66)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pick the best free model for a task")
    ap.add_argument("task", nargs="?", help="Task description (or pipe it on stdin)")
    ap.add_argument("--json", action="store_true", help="Emit the decision as JSON")
    ap.add_argument("--tag", "-t", help="Force a task type instead of classifying")
    ap.add_argument("--config-dir", help="Directory holding the routing-*.json files")
    ap.add_argument("--state-path", help="Override the state file location")
    ap.add_argument("--no-record", action="store_true", help="Do not write to the state file")
    ap.add_argument("--ignore-keys", action="store_true",
                    help="Consider providers even when no API key is configured")
    ap.add_argument("--list", action="store_true", help="List providers and models, then exit")
    ap.add_argument("--doctor", action="store_true", help="Check config, keys and routing health")
    ap.add_argument("--live", action="store_true",
                    help="With --doctor, verify catalog model ids against the provider APIs")
    args = ap.parse_args(argv)

    try:
        router = Router(config_dir=args.config_dir, state_path=args.state_path)
    except (FileNotFoundError, json.JSONDecodeError, ConfigVersionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.doctor:
        return doctor(router, live=args.live)

    if args.list:
        for provider in router.catalog["provider_order"]:
            pdef = router.catalog["providers"][provider]
            configured = "configured" if provider_is_configured(pdef, router.keys) else "NO KEY"
            print(f"\n[{provider}] free={pdef.get('free')} {configured}")
            for m in pdef["models"]:
                print(f"  {m['class']:<7} {m['id']:<42} ctx={m.get('context'):<8} "
                      f"{','.join(m.get('strengths', []))}")
        return 0

    text = read_task(args.task)
    decision = router.route(text, tag=args.tag, require_keys=not args.ignore_keys)

    if not args.no_record:
        router.record_decision(decision, text)
        try:
            router.save_state()
        except OSError as exc:
            print(f"WARNING: could not write state: {exc}", file=sys.stderr)

    if args.json:
        print(json.dumps(decision, indent=2, ensure_ascii=False))
    else:
        print_human(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
