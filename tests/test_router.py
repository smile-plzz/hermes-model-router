"""Tests for the model router. Run: python -m pytest tests -q"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import providers as provider_api  # noqa: E402
from respond import respond  # noqa: E402
from router import Router, classify, config_search_path, resolve_config_dir  # noqa: E402

ALL_KEYS = {
    "GROQ_API_KEY": "k", "GOOGLE_API_KEY": "k",
    "MISTRAL_API_KEY": "k", "OPENROUTER_API_KEY": "k",
}


@pytest.fixture
def router(tmp_path):
    return Router(config_dir=REPO, state_path=tmp_path / "state.json", keys=dict(ALL_KEYS))


# -- classification ---------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("hey what's up", "simple"),
    ("thanks", "simple"),
    ("write a python function that merges two sorted lists", "code"),
    ("fix this bug in my python code", "code"),
    ("debug my React component that's not rendering", "code"),
    ("analyze the pros and cons of microservices vs monolith", "reasoning"),
    ("write a 4-line poem about rain in Dhaka", "creative"),
    ("brainstorm 10 AI startup ideas", "creative"),
    ("summarize this article in one sentence", "summary"),
    ("summarize this 200-page document", "long_context"),
])
def test_classification(router, text, expected):
    assert router.route(text)["task_type"] == expected


def test_python_hello_world_is_code_not_simple(router):
    """Regression: this tied simple/code at 10 and dict order sent it to an 8b model."""
    decision = router.route("write a one-line python hello world")
    assert decision["task_type"] == "code"
    assert decision["class"] == "large"


def test_long_message_is_never_simple(router):
    """`simple` is gated on length, so a greeting prefix cannot downgrade real work."""
    text = ("hey thanks for that, now please refactor the authentication module "
            "so the session tokens are stored server side")
    assert router.route(text)["task_type"] == "code"


def test_precedence_breaks_ties_deterministically(router):
    """Equal scores must not depend on dict insertion order."""
    result = classify("summarize the code", router.rules)
    assert result["scores"]["code"] == result["scores"]["summary"]
    assert result["best"] == "code"  # code has the higher declared precedence


def test_prefix_matching_does_not_overreach(router):
    hits = classify("debugging the parser", router.rules)["matched"]["code"]
    assert "debug" in hits
    # "large" is a long_context keyword; "enlarge" must not trigger it.
    assert "long_context" not in classify("enlarge the image", router.rules)["scores"]


def test_unmatched_text_asks_the_user(router):
    decision = router.route("zqx wobble frobnicate")
    assert decision["ask_user"] is True
    assert decision["task_type"] == "simple"


def test_tag_overrides_classification(router):
    decision = router.route("write a poem about rain", tag="code")
    assert decision["task_type"] == "code"
    assert decision["confidence"] == 100


def test_unknown_tag_falls_back_to_classification(router):
    decision = router.route("write a poem about rain", tag="nonsense")
    assert decision["task_type"] == "creative"


# -- candidate scoring ------------------------------------------------------
def test_long_context_requires_a_big_window(router):
    decision = router.route("summarize this 200-page document")
    context = decision["candidates"][0]["context"]
    assert context >= router.rules["task_types"]["long_context"]["min_context"]


def test_code_task_picks_a_code_capable_model(router):
    decision = router.route("write a python function to merge two sorted lists")
    model = next(m for m in router.catalog["providers"][decision["provider"]]["models"]
                 if m["id"] == decision["model"])
    assert "code" in model["strengths"]


def test_ties_fall_back_to_catalog_order_not_alphabetical(router):
    """groq/compound-mini sorts before llama-3.1-8b-instant alphabetically."""
    decision = router.route("thanks")
    assert decision["model"] == "llama-3.1-8b-instant"


def test_candidates_are_sorted_by_score(router):
    scores = [c["score"] for c in router.route("write a python function")["candidates"]]
    assert scores == sorted(scores, reverse=True)


# -- provider availability --------------------------------------------------
def test_providers_without_keys_are_skipped(tmp_path):
    router = Router(config_dir=REPO, state_path=tmp_path / "s.json",
                    keys={"MISTRAL_API_KEY": "k"})
    decision = router.route("write a python function to merge two sorted lists")
    # nous needs no key, mistral has one; groq/gemini/openrouter must be absent.
    assert {c["provider"] for c in decision["candidates"]} <= {"mistral", "nous"}
    assert decision["provider"] == "mistral"


def test_cooldown_removes_a_provider(router):
    router.provider_state("groq")["cooldown_until"] = (
        datetime.now(timezone.utc) + timedelta(minutes=5)
    ).isoformat()
    decision = router.route("write a python function to merge two sorted lists")
    assert decision["provider"] != "groq"
    assert all(c["provider"] != "groq" for c in decision["candidates"])


def test_expired_cooldown_is_ignored(router):
    router.provider_state("groq")["cooldown_until"] = (
        datetime.now(timezone.utc) - timedelta(minutes=5)
    ).isoformat()
    assert router.route("write a python function to merge two sorted lists")["provider"] == "groq"


def test_malformed_cooldown_does_not_crash(router):
    router.provider_state("groq")["cooldown_until"] = "not-a-timestamp"
    assert router.cooling_down("groq") is False


def test_naive_cooldown_timestamp_is_treated_as_utc(router):
    """The old state file wrote naive timestamps; comparing them must not raise."""
    router.provider_state("groq")["cooldown_until"] = (
        datetime.now(timezone.utc) + timedelta(minutes=5)
    ).replace(tzinfo=None).isoformat()
    assert router.cooling_down("groq") is True


def test_routing_survives_every_provider_being_unusable(tmp_path):
    router = Router(config_dir=REPO, state_path=tmp_path / "s.json", keys={})
    for name in router.catalog["providers"]:
        router.provider_state(name)["cooldown_until"] = (
            datetime.now(timezone.utc) + timedelta(minutes=5)
        ).isoformat()
    decision = router.route("write a python function")
    assert decision["provider"] is None
    assert decision["ask_user"] is True


# -- state ------------------------------------------------------------------
def test_rate_limits_accumulate_into_a_cooldown(router):
    threshold = router.state["config"]["cooldown_after_rate_limits"]
    for _ in range(threshold):
        router.record_rate_limit("groq")
    assert router.cooling_down("groq") is True


def test_success_clears_the_rate_limit_counter(router):
    router.record_rate_limit("groq")
    router.record_success("groq")
    assert router.provider_state("groq")["rate_limit_hits"] == 0
    assert router.provider_state("groq")["cooldown_until"] is None


def test_model_switches_only_count_actual_switches(router):
    """Regression: the old code compared against a key that never existed, so
    every task counted as a switch (85 tasks / 85 switches in production)."""
    for text in ("thanks", "cheers", "write a python function"):
        router.record_decision(router.route(text), text)
    assert router.state["session"]["tasks"] == 3
    assert router.state["session"]["model_switches"] == 2


def test_state_round_trips_and_heals_missing_providers(tmp_path):
    state_file = tmp_path / "s.json"
    state_file.write_text(json.dumps({"version": 2, "providers": {"groq": {}}}), encoding="utf-8")
    router = Router(config_dir=REPO, state_path=state_file, keys=dict(ALL_KEYS))
    assert set(router.state["providers"]) == set(router.catalog["providers"])
    router.record_rate_limit("mistral")
    router.save_state()
    assert json.loads(state_file.read_text(encoding="utf-8"))["providers"]["mistral"]["rate_limit_hits"] == 1


def test_corrupt_state_file_is_recoverable(tmp_path):
    state_file = tmp_path / "s.json"
    state_file.write_text("{not json", encoding="utf-8")
    router = Router(config_dir=REPO, state_path=state_file, keys=dict(ALL_KEYS))
    assert router.route("thanks")["provider"] is not None


# -- config discovery -------------------------------------------------------
def test_config_resolves_to_the_repo_without_any_env(monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_ROUTER_HOME", raising=False)
    assert resolve_config_dir() == REPO


def test_explicit_dir_outranks_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert config_search_path(REPO)[0] == REPO


def test_bad_explicit_config_dir_fails_loudly(tmp_path):
    """A typo in --config-dir must not silently fall through to the bundled config."""
    with pytest.raises(FileNotFoundError, match="routing-catalog.json"):
        resolve_config_dir(tmp_path / "nowhere")


def test_env_pointing_somewhere_empty_falls_back_to_the_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_ROUTER_HOME", raising=False)
    assert resolve_config_dir() == REPO


# -- error classification ---------------------------------------------------
@pytest.mark.parametrize("status,body,expected", [
    (401, "", "auth"),
    (403, "", "auth"),
    (429, "", "rate_limit"),
    (404, "", "bad_model"),
    (503, "", "transient"),
    (400, "API key not valid. Please pass a valid API key.", "auth"),
    (400, "You exceeded your current quota", "rate_limit"),
    (400, "model gpt-nope does not exist", "bad_model"),
    (400, "something else entirely", "error"),
])
def test_http_status_classification(status, body, expected):
    assert provider_api._classify_status(status, body) == expected


# -- end-to-end failover ----------------------------------------------------
def test_failover_walks_providers_and_records_state(router, monkeypatch):
    calls: list[str] = []

    def fake_call(provider_def, model, prompt, api_key, max_tokens=2048, timeout=45):
        calls.append(provider_def["endpoint"] if "endpoint" in provider_def else "nous")
        if len(calls) == 1:
            return provider_api.Result("rate_limit", message="slow down", status=429)
        return provider_api.Result("ok", text="answer", status=200)

    monkeypatch.setattr(provider_api, "call", fake_call)
    result = respond(router, "write a python function to merge two sorted lists")

    assert result["success"] is True
    assert result["response"] == "answer"
    assert [a["outcome"] for a in result["attempts"]] == ["rate_limit", "ok"]
    assert router.provider_state(result["attempts"][0]["provider"])["rate_limit_hits"] == 1


def test_failover_tries_one_model_per_provider(router, monkeypatch):
    seen: list[str] = []

    def always_fail(provider_def, model, prompt, api_key, max_tokens=2048, timeout=45):
        seen.append(model)
        return provider_api.Result("transient", message="boom")

    monkeypatch.setattr(provider_api, "call", always_fail)
    result = respond(router, "write a python function", max_attempts=4)

    assert result["success"] is False
    providers_tried = [a["provider"] for a in result["attempts"]]
    assert len(providers_tried) == len(set(providers_tried)) == 4
