#!/usr/bin/env python3
"""Route a task to the best free model, call it, and return the answer.

Walks the ranked candidate list on failure and feeds what happened back into
routing-state.json, so a rate-limited provider actually cools down.

  python respond.py "write a python function that merges two sorted lists"
  python respond.py "analyze microservices vs monolith" --json
  echo "hello" | python respond.py --raw
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import providers as provider_api
from router import ConfigVersionError, Router, provider_key, read_task

# One model per provider is enough of a retry budget; beyond that we are just
# hammering the same rate limit with a slightly smaller model.
DEFAULT_MAX_ATTEMPTS = 4


def _attempt_order(candidates: list[dict], max_attempts: int) -> list[dict]:
    """Best candidate first, then the best remaining model of each other provider."""
    order: list[dict] = []
    seen_providers: set[str] = set()
    for cand in candidates:
        if cand["provider"] in seen_providers:
            continue
        seen_providers.add(cand["provider"])
        order.append(cand)
        if len(order) >= max_attempts:
            break
    return order


def respond(router: Router, text: str, tag: str | None = None,
            max_tokens: int = 2048, timeout: int = provider_api.DEFAULT_TIMEOUT,
            max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> dict:
    started = time.time()
    decision = router.route(text, tag=tag)
    attempts: list[dict] = []

    for cand in _attempt_order(decision["candidates"], max_attempts):
        provider = cand["provider"]
        pdef = router.catalog["providers"][provider]
        key = provider_key(pdef, router.keys) or ""
        call_started = time.time()
        result = provider_api.call(pdef, cand["model"], text, key,
                                   max_tokens=max_tokens, timeout=timeout)
        elapsed = int((time.time() - call_started) * 1000)
        attempts.append({
            "provider": provider,
            "model": cand["model"],
            "outcome": result.kind,
            "status": result.status,
            "message": result.message,
            "latency_ms": elapsed,
        })

        if result.ok:
            router.record_success(provider)
            return {
                "success": True,
                "task_type": decision["task_type"],
                "provider": provider,
                "model": cand["model"],
                "response": result.text,
                "attempts": attempts,
                "routing": decision,
                "latency_ms": int((time.time() - started) * 1000),
            }

        if result.kind == "rate_limit":
            router.record_rate_limit(provider)
        else:
            router.record_failure(provider)

    return {
        "success": False,
        "task_type": decision["task_type"],
        "provider": None,
        "model": None,
        "response": "",
        "error": "every candidate failed",
        "attempts": attempts,
        "routing": decision,
        "latency_ms": int((time.time() - started) * 1000),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Route a task to the best free model and answer it")
    ap.add_argument("task", nargs="?", help="Task description (or pipe it on stdin)")
    ap.add_argument("--json", action="store_true", help="Emit the full result as JSON")
    ap.add_argument("--raw", action="store_true", help="Print only the model's answer")
    ap.add_argument("--tag", "-t", help="Force a task type instead of classifying")
    ap.add_argument("--config-dir", help="Directory holding the routing-*.json files")
    ap.add_argument("--state-path", help="Override the state file location")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=provider_api.DEFAULT_TIMEOUT)
    ap.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    ap.add_argument("--no-record", action="store_true", help="Do not write to the state file")
    args = ap.parse_args(argv)

    try:
        router = Router(config_dir=args.config_dir, state_path=args.state_path)
    except (FileNotFoundError, json.JSONDecodeError, ConfigVersionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    text = read_task(args.task)
    result = respond(router, text, tag=args.tag, max_tokens=args.max_tokens,
                     timeout=args.timeout, max_attempts=args.max_attempts)

    if not args.no_record:
        router.record_decision(result["routing"], text)
        try:
            router.save_state()
        except OSError as exc:
            print(f"WARNING: could not write state: {exc}", file=sys.stderr)

    if args.raw:
        print(result["response"] or result.get("error", ""), end="\n" if result["response"] else "")
    elif args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("=" * 66)
        if result["success"]:
            print(f"TYPE     : {result['task_type']}")
            print(f"ANSWERED : {result['provider']}/{result['model']}  ({result['latency_ms']}ms)")
            for a in result["attempts"][:-1]:
                print(f"  fell back from {a['provider']}/{a['model']}: {a['outcome']} — {a['message'][:80]}")
            print("-" * 66)
            print(result["response"])
        else:
            print(f"FAILED after {len(result['attempts'])} attempt(s) ({result['latency_ms']}ms)")
            for a in result["attempts"]:
                print(f"  {a['provider']}/{a['model']}: {a['outcome']} — {a['message'][:100]}")
        print("=" * 66)

    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
