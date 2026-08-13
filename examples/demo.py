#!/usr/bin/env python3
"""Run a batch of sample tasks through the router and print the decisions.

  python examples/demo.py            # routing decisions only, no API calls
  python examples/demo.py --live     # also call each model and time it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from router import Router  # noqa: E402

TASKS = [
    "hey what's up",
    "thanks, that worked",
    "write a python function that merges two sorted lists",
    "fix this bug in my python code",
    "debug my React component that's not rendering",
    "write a one-line python hello world",
    "analyze the pros and cons of microservices vs monolith",
    "plan a 3-day trip to Singapore",
    "write a 4-line poem about rain in Dhaka",
    "brainstorm 10 AI startup ideas for 2026",
    "summarize this article in one sentence",
    "summarize this 200-page document",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Model router demo")
    ap.add_argument("--live", action="store_true", help="Actually call the chosen models")
    args = ap.parse_args()

    router = Router(config_dir=REPO, state_path=Path(__file__).parent / ".demo-state.json")
    print(f"config: {router.config_dir}\n")
    header = f"{'TASK':<52} {'TYPE':<13} {'CLASS':<7} PROVIDER/MODEL"
    print(header)
    print("-" * len(header))

    for task in TASKS:
        d = router.route(task)
        flag = "  <ask>" if d["ask_user"] else ""
        print(f"{task[:50]:<52} {d['task_type']:<13} {d.get('class', '-'):<7} "
              f"{d['provider']}/{d['model']}{flag}")

    if args.live:
        import respond as respond_mod
        print("\nLive calls:")
        for task in TASKS[:4]:
            result = respond_mod.respond(router, task, max_tokens=128)
            status = "ok" if result["success"] else "FAILED"
            print(f"  {task[:44]:<46} {status:<7} {result['provider']}/{result['model']} "
                  f"{result['latency_ms']}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
