#!/usr/bin/env python3
"""HTTP adapters for the model providers in routing-catalog.json.

Every call returns a Result. `kind` is what the router needs in order to react:
  ok         - `text` holds the completion
  auth       - key missing or rejected; do not retry this provider
  rate_limit - back off, feeds the cooldown counter
  transient  - network/5xx; the next candidate is tried
  bad_model  - the model id is not valid for this provider
  error      - anything else
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

import requests

DEFAULT_TIMEOUT = 45
_REFERER = "https://github.com/smile-plzz/hermes-model-router"


@dataclass
class Result:
    kind: str
    text: str = ""
    message: str = ""
    status: int | None = None
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.kind == "ok"


_AUTH_HINTS = ("api key not valid", "invalid api key", "invalid_api_key",
               "unauthorized", "authentication", "api key expired")
_QUOTA_HINTS = ("quota", "rate limit", "resource_exhausted", "too many requests")


def _classify_status(status: int, body: str) -> str:
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "rate_limit"
    if status == 404:
        return "bad_model"
    if status >= 500:
        return "transient"
    if status == 400:
        # Google answers 400 for bad keys and unknown models alike.
        low = body.lower()
        if any(h in low for h in _AUTH_HINTS):
            return "auth"
        if any(h in low for h in _QUOTA_HINTS):
            return "rate_limit"
        if "model" in low and ("not found" in low or "not supported" in low or "does not exist" in low):
            return "bad_model"
    return "error"


def _error_text(payload: object, fallback: str) -> str:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err:
            return str(err)
        if payload.get("message"):
            return str(payload["message"])
    return fallback[:300]


def _request(url: str, *, headers: dict, json_body: dict, timeout: int) -> Result | requests.Response:
    try:
        return requests.post(url, headers=headers, json=json_body, timeout=timeout)
    except requests.Timeout:
        return Result("transient", message=f"timeout after {timeout}s")
    except requests.RequestException as exc:
        return Result("transient", message=str(exc))


def _finish(resp: requests.Response, extract) -> Result:
    try:
        payload = resp.json()
    except ValueError:
        payload = None
    if resp.status_code != 200:
        return Result(
            _classify_status(resp.status_code, resp.text),
            message=_error_text(payload, resp.text),
            status=resp.status_code,
        )
    if payload is None:
        return Result("error", message=f"non-JSON response: {resp.text[:200]}", status=200)
    text = extract(payload)
    if not text:
        return Result("error", message=_error_text(payload, str(payload)[:300]), status=200,
                      raw=payload if isinstance(payload, dict) else {})
    return Result("ok", text=text, status=200, raw=payload)


def _extract_openai(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content") or ""


def _extract_gemini(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts)


def call_openai_chat(provider_def: dict, model: str, prompt: str, api_key: str,
                     max_tokens: int = 2048, timeout: int = DEFAULT_TIMEOUT) -> Result:
    if not api_key:
        return Result("auth", message="no API key configured")
    resp = _request(
        provider_def["endpoint"],
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": _REFERER,
            "X-Title": "hermes-model-router",
        },
        json_body={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
        timeout=timeout,
    )
    return resp if isinstance(resp, Result) else _finish(resp, _extract_openai)


def call_gemini(provider_def: dict, model: str, prompt: str, api_key: str,
                max_tokens: int = 2048, timeout: int = DEFAULT_TIMEOUT) -> Result:
    if not api_key:
        return Result("auth", message="no API key configured")
    url = provider_def["endpoint"].replace("{model}", model)
    resp = _request(
        url,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        json_body={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7},
        },
        timeout=timeout,
    )
    return resp if isinstance(resp, Result) else _finish(resp, _extract_gemini)


def call_hermes_cli(provider_def: dict, model: str, prompt: str, api_key: str,
                    max_tokens: int = 2048, timeout: int = DEFAULT_TIMEOUT) -> Result:
    """Local Hermes CLI — the Nous path, which authenticates via its own OAuth."""
    cwd = os.environ.get("HERMES_HOME") or None
    try:
        proc = subprocess.run(
            ["hermes", "chat", "-q", prompt, "--quiet"],
            capture_output=True, text=True, timeout=timeout, cwd=cwd,
        )
    except FileNotFoundError:
        return Result("auth", message="hermes CLI not on PATH")
    except subprocess.TimeoutExpired:
        return Result("transient", message=f"hermes CLI timeout after {timeout}s")
    noise = ("session_id:", "HTTP", "Warning:", "⚠")
    lines = [l for l in proc.stdout.splitlines() if l.strip() and not l.startswith(noise)]
    text = "\n".join(lines).strip()
    if proc.returncode != 0 or not text:
        return Result("error", message=(proc.stderr or "empty response")[:300])
    return Result("ok", text=text)


_DISPATCH = {
    "openai_chat": call_openai_chat,
    "gemini": call_gemini,
    "hermes_cli": call_hermes_cli,
}


def call(provider_def: dict, model: str, prompt: str, api_key: str,
         max_tokens: int = 2048, timeout: int = DEFAULT_TIMEOUT) -> Result:
    handler = _DISPATCH.get(provider_def.get("api", ""))
    if handler is None:
        return Result("error", message=f"unsupported api '{provider_def.get('api')}'")
    return handler(provider_def, model, prompt, api_key, max_tokens, timeout)


def list_models(provider_def: dict, api_key: str, timeout: int = 25) -> tuple[set[str], str | None]:
    """Return (live model ids, error). Used by `router.py --doctor --live`."""
    url = provider_def.get("models_endpoint")
    if not url:
        return set(), "no models endpoint"
    headers = {}
    if provider_def.get("api") == "gemini":
        headers["x-goog-api-key"] = api_key
        url += "?pageSize=200"
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        return set(), str(exc)
    if resp.status_code != 200:
        return set(), f"HTTP {resp.status_code}: {resp.text[:120]}"
    try:
        payload = resp.json()
    except ValueError:
        return set(), "non-JSON model list"
    entries = payload.get("data") or payload.get("models") or []
    ids = set()
    for entry in entries:
        raw = entry.get("id") or entry.get("name") or ""
        if raw:
            ids.add(raw.split("models/")[-1])
    return ids, None
