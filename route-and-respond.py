#!/usr/bin/env python3
"""
route-and-respond.py — Classify a task, pick the best free model, call its API, return the response.

Combines routing-router.py (classification + scoring) with direct API calls to:
  Groq  → api.groq.com        (gsk_ key)
  Gemini → generativelanguage.googleapis.com (AQ_ key)
  Mistral → api.mistral.ai    (fTgK_ key)
  OpenRouter → openrouter.ai  (sk-or-v1_ key)
  Nous/Solar → hermes chat -q fallback

Usage:
  python route-and-respond.py "write a function to merge sorted lists"
  python route-and-respond.py "analyze microservices" --json
  echo "hello" | python route-and-respond.py --json
  python route-and-respond.py "task" --tag code --json
"""

import argparse, json, os, subprocess, sys, time
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

HERMES_HOME = os.environ.get("HERMES_HOME") or str(Path.home() / "AppData" / "Local" / "hermes")
ROUTER = Path(HERMES_HOME) / "routing-router.py"


# ── Load API keys from .env ────────────────────────────────────────────────

def load_keys():
    env_path = Path(HERMES_HOME) / ".env"
    keys = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip()
    return keys


# ── API call wrappers ───────────────────────────────────────────────────────

def _parse_chat_completion(resp):
    """Parse OpenAI-compatible response (Groq, Mistral, OpenRouter)."""
    try:
        data = resp.json()
    except Exception:
        return {"text": "", "error": True, "error_message": f"JSON parse: {resp.text[:200]}"}
    if "choices" in data and data["choices"]:
        return {"text": data["choices"][0]["message"]["content"], "error": False}
    if "error" in data:
        return {"text": "", "error": True,
                "error_message": data["error"].get("message", str(data["error"]))}
    return {"text": "", "error": True, "error_message": f"Unexpected: {str(data)[:200]}"}

def _parse_gemini(resp):
    """Parse Google Gemini generateContent response."""
    try:
        data = resp.json()
    except Exception:
        return {"text": "", "error": True, "error_message": f"JSON parse: {resp.text[:200]}"}
    if "candidates" in data and data["candidates"]:
        parts = data["candidates"][0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        return {"text": text, "error": False}
    if "error" in data:
        return {"text": "", "error": True,
                "error_message": data["error"].get("message", str(data["error"]))}
    return {"text": "", "error": True, "error_message": f"Unexpected: {str(data)[:200]}"}

def call_groq(model_id, task_text, api_key, max_tokens=4000):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://hermes-agent.nousresearch.com"}
    payload = {"model": model_id, "messages": [{"role": "user", "content": task_text}],
               "max_tokens": max_tokens, "temperature": 0.7}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        return _parse_chat_completion(r)
    except Exception as e:
        return {"text": "", "error": True, "error_message": str(e)}

def call_gemini(model_id, task_text, api_key, max_tokens=4000):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}"
           f":generateContent?key={api_key}")
    payload = {"contents": [{"parts": [{"text": task_text}]}],
               "generationConfig": {"maxOutputTokens": max_tokens}}
    try:
        r = requests.post(url, json=payload, timeout=30)
        return _parse_gemini(r)
    except Exception as e:
        return {"text": "", "error": True, "error_message": str(e)}

def call_mistral(model_id, task_text, api_key, max_tokens=4000):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model_id, "messages": [{"role": "user", "content": task_text}],
               "max_tokens": max_tokens, "temperature": 0.7}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        return _parse_chat_completion(r)
    except Exception as e:
        return {"text": "", "error": True, "error_message": str(e)}

def call_openrouter(model_id, task_text, api_key, max_tokens=4000):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://hermes-agent.nousresearch.com"}
    payload = {"model": model_id, "messages": [{"role": "user", "content": task_text}],
               "max_tokens": max_tokens, "temperature": 0.7}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        return _parse_chat_completion(r)
    except Exception as e:
        return {"text": "", "error": True, "error_message": str(e)}

def call_nous(task_text, max_tokens=4000):
    """Fallback: hermes chat -q with default Nous model."""
    try:
        r = subprocess.run(
            ["hermes", "chat", "-q", task_text, "--quiet"],
            capture_output=True, text=True, timeout=60, cwd=HERMES_HOME,
        )
        lines = r.stdout.strip().split("\n")
        response_lines = [l for l in lines
                          if not l.startswith("session_id:")
                          and not l.startswith("HTTP")
                          and not l.startswith("⚠")
                          and not l.startswith("Warning:")
                          and l.strip()]
        text = "\n".join(response_lines)
        if not text or r.returncode != 0:
            return {"text": "", "error": True,
                    "error_message": r.stderr[:200] if r.stderr else "empty response"}
        return {"text": text, "error": False}
    except Exception as e:
        return {"text": "", "error": True, "error_message": str(e)}


# ── Main: classify → call → return ─────────────────────────────────────────

def route_and_respond(task_text, keys, tag=None):
    start = time.time()

    # 1. Classify + pick (via routing-router.py)
    router_cmd = ["python3", str(ROUTER), "--json"]
    if tag:
        router_cmd.extend(["--tag", tag])
    try:
        r = subprocess.run(router_cmd, input=task_text,
                           capture_output=True, text=True, timeout=30, cwd=HERMES_HOME)
        if r.returncode != 0:
            return {"success": False, "response": f"Router error: {r.stderr[:200]}",
                    "provider": None, "model": None, "latency_ms": 0}
        decision = json.loads(r.stdout)
    except Exception as e:
        return {"success": False, "response": f"Router exception: {e}",
                "provider": None, "model": None, "latency_ms": 0}

    # 2. Try each candidate in score order
    messages = [{"role": "user", "content": task_text}]
    errors = []

    for cand in decision.get("all_candidates", []):
        prov = cand["provider"]
        model = cand["model"]

        if prov == "groq":
            api_key = keys.get("GROQ_API_KEY", "")
            if api_key:
                resp = call_groq(model, task_text, api_key)
        elif prov == "gemini":
            api_key = keys.get("GOOGLE_API_KEY") or keys.get("GEMINI_API_KEY", "")
            if api_key:
                resp = call_gemini(model, task_text, api_key)
        elif prov == "mistral":
            api_key = keys.get("MISTRAL_API_KEY", "")
            if api_key:
                resp = call_mistral(model, task_text, api_key)
        elif prov == "openrouter":
            api_key = keys.get("OPENROUTER_API_KEY", "")
            if api_key:
                resp = call_openrouter(model, task_text, api_key)
        elif prov == "nous":
            resp = call_nous(task_text)
        else:
            resp = {"text": "", "error": True, "error_message": f"Unknown provider: {prov}"}

        if not resp.get("error"):
            elapsed = int((time.time() - start) * 1000)
            return {"success": True,
                    "routing": decision,
                    "response": resp["text"],
                    "provider": prov,
                    "model": model,
                    "latency_ms": elapsed}
        else:
            errors.append(f"{prov}/{model}: {resp.get('error_message', 'failed')}")

    elapsed = int((time.time() - start) * 1000)
    return {"success": False,
            "routing": decision,
            "response": f"All providers failed. Errors:\n" + "\n".join(errors),
            "provider": None, "model": None, "latency_ms": elapsed,
            "errors": errors}


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Route a task to the best free model and respond")
    ap.add_argument("task", nargs="?", help="Task description")
    ap.add_argument("--json", action="store_true", help="JSON output (default: human-readable)")
    ap.add_argument("--raw", action="store_true", help="Print response text only")
    ap.add_argument("--tag", "-t", help="Explicit task type (simple/code/reasoning/creative/summary/long_context)")
    args = ap.parse_args()

    task_text = args.task
    if not task_text:
        if not sys.stdin.isatty():
            task_text = sys.stdin.read().strip()
        else:
            print("Usage: python route-and-respond.py \"your task\"", file=sys.stderr)
            sys.exit(1)

    keys = load_keys()
    result = route_and_respond(task_text, keys, tag=args.tag)

    if args.raw:
        print(result.get("response", ""))
    elif args.json:
        out = dict(result)
        out["success"] = result.get("success", False)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        # Human-readable
        print("=" * 60)
        if result.get("success"):
            print(f"📋 TASK: {task_text[:100]}")
            print(f"🎯 PROVIDER: {result.get('provider')}")
            print(f"🤖 MODEL: {result.get('model')}")
            print(f"⏱️  LATENCY: {result.get('latency_ms')}ms")
            print("─" * 60)
            print(result.get("response", ""))
            print("=" * 60)
        else:
            print(f"❌ FAILED ({result.get('latency_ms')}ms)")
            print(result.get("response", "Unknown error"))
            print("=" * 60)


if __name__ == "__main__":
    main()
