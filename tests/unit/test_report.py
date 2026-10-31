from __future__ import annotations

from datetime import UTC, datetime

from agent.config import Settings
from evals.runner import TrialResult, render_markdown, summarise, wilson_interval


def test_interval_narrows_as_the_sample_grows() -> None:
    tiny = wilson_interval(3, 3)
    large = wilson_interval(75, 75)
    assert tiny[0] < large[0] < 1.0
    assert large[0] > 0.9
    assert large[1] == 1.0


def test_interval_handles_an_empty_sample() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_interval_never_leaves_the_unit_range() -> None:
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert 0.0 < high < 0.4
    low, high = wilson_interval(10, 10)
    assert low > 0.6
    assert high == 1.0


def _trial(scenario: str, passed: bool, trial: int = 0) -> TrialResult:
    return TrialResult(
        scenario_id=scenario,
        category="read",
        trial=trial,
        passed=passed,
        turns=[],
        failures=[] if passed else ["nope"],
    )


def test_summary_carries_intervals_and_counts() -> None:
    results = [_trial("a", True), _trial("a", True), _trial("b", True), _trial("b", False)]
    summary = summarise(results, trials=2)
    assert summary["trial_runs"] == 4
    assert summary["tsr_pass_1"] == 0.75
    assert summary["tsr_ci"][0] < 0.75 < summary["tsr_ci"][1]
    assert summary["pass_k"] == 0.5
    assert summary["categories"]["read"]["pass_1_ci"][0] < 0.75


def test_report_states_provenance_and_the_mode_caveat() -> None:
    results = [_trial("a", True)]
    summary = summarise(results, trials=1)
    settings = Settings(
        _env_file=None,
        database_url="postgresql://u:p@localhost:5432/db",
        customer_api_keys={"k": "11111111-1111-4111-8111-111111111111"},
        llm_model="gpt-test",
        llm_temperature=0.0,
    )
    markdown = render_markdown(
        summary=summary,
        results=results,
        fingerprint={"commit": "abc123", "dirty_worktree": False, "scenario_sha256": "f" * 64},
        settings=settings,
        mode="live",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert "`abc123`" in markdown
    assert "`gpt-test`" in markdown
    assert "95% CI" in markdown
    assert "| Trial runs (n) | 1 |" in markdown
    assert "prompt version" in markdown.lower()
    assert "Live mode calls the configured provider" in markdown
