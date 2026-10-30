"""Run the graded scenario suite and write the task-success report.

    python scripts/evaluate.py --trials 3            # offline replay, no API key
    python scripts/evaluate.py --mode live --trials 3

Offline mode replays scripted model plans so the pipeline is deterministic.
Live mode calls the configured provider and measures the model itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.config import Settings
from agent.db.session import create_engine
from agent.errors import ModelUnavailable
from evals.runner import (
    TrialResult,
    _prepare_schema,
    config_fingerprint,
    ensure_database,
    render_markdown,
    run_trial,
    summarise,
)
from evals.scenarios import SCENARIOS
from evals.scripted import ScriptedModel

DEFAULT_EVAL_URL = "postgresql+psycopg://postgres:postgres@127.0.0.1:55432/pm_eval"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3, help="runs per scenario (k in pass^k)")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--scenario", action="append", dest="scenarios", help="limit to these ids")
    parser.add_argument("--out", default="evals/report", help="directory for report.md/report.json")
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.95,
        help="exit non-zero when pass^1 is below this value",
    )
    return parser.parse_args(argv)


def _settings(args: argparse.Namespace) -> Settings:
    base = Settings()  # type: ignore[call-arg]
    database_url = args.database_url or DEFAULT_EVAL_URL
    return base.model_copy(
        update={
            "database_url": database_url,
            "checkpoint_dsn": None,
            "environment": "eval",
            "rate_limit_per_min": 100_000,
            "rate_limit_burst": 100_000,
        }
    )


@dataclass(frozen=True)
class Artifacts:
    summary: dict[str, Any]
    markdown: str
    payload: dict[str, Any]


async def _run(args: argparse.Namespace) -> Artifacts | int:
    settings = _settings(args)
    ensure_database(settings.database_url)

    selected = SCENARIOS
    if args.scenarios:
        wanted = set(args.scenarios)
        selected = tuple(scenario for scenario in SCENARIOS if scenario.id in wanted)
        missing = wanted - {scenario.id for scenario in SCENARIOS}
        if missing:
            print(f"unknown scenario id(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 2

    live_model = None
    if args.mode == "live":
        from agent.agent.llm import build_chat_model

        try:
            live_model = build_chat_model(settings)
        except ModelUnavailable as exc:
            print(f"live mode needs LLM_MODEL and OPENAI_API_KEY: {exc}", file=sys.stderr)
            return 2

    engine = create_engine(settings.database_url)
    await _prepare_schema(engine)

    results: list[TrialResult] = []
    try:
        for scenario in selected:
            plan = [message for turn in scenario.turns for message in turn.plan]
            for trial in range(args.trials):
                model = live_model or ScriptedModel(responses=plan)
                result = await run_trial(
                    scenario, settings=settings, model=model, engine=engine, trial=trial
                )
                results.append(result)
                status = "PASS" if result.passed else "FAIL"
                print(f"[{status}] {scenario.id} (trial {trial + 1}/{args.trials})")
                for failure in result.failures:
                    print(f"        - {failure}")
    finally:
        await engine.dispose()

    summary = summarise(results, args.trials)
    generated_at = datetime.now(UTC)
    fingerprint = config_fingerprint(Path("evals/scenarios.py"))
    markdown = render_markdown(
        summary=summary,
        results=results,
        fingerprint=fingerprint,
        settings=settings,
        mode=args.mode,
        generated_at=generated_at,
    )
    payload = {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "mode": args.mode,
        "config": fingerprint,
        "summary": summary,
        "trials": [
            {
                "scenario": result.scenario_id,
                "category": result.category,
                "trial": result.trial,
                "passed": result.passed,
                "failures": result.failures,
                "turns": [turn.as_dict() for turn in result.turns],
            }
            for result in results
        ],
    }
    return Artifacts(summary=summary, markdown=markdown, payload=payload)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = _parse_args(argv)
    try:
        outcome = asyncio.run(_run(args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    if isinstance(outcome, int):
        return outcome

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(outcome.markdown, encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(outcome.payload, indent=2), encoding="utf-8")

    summary = outcome.summary
    print("")
    print(f"pass^1 = {summary['tsr_pass_1']:.1%}  pass^{args.trials} = {summary['pass_k']:.1%}")
    print(f"latency p50/p95 = {summary['latency_ms']['p50']} / {summary['latency_ms']['p95']} ms")
    print(f"report written to {out_dir / 'report.md'}")
    return 0 if summary["tsr_pass_1"] >= args.fail_under else 1


if __name__ == "__main__":
    raise SystemExit(main())
