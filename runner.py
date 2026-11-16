"""CLI runner: tasks x policies x seeds -> traces + summary.csv.

Owner: 丙 (policies & experiments)
summary.csv is the team experiment log (one row per run, appended).

Smoke test (no API needed):
  python runner.py --tasks tasks/dev.json --policies c0 c1 c2 --seeds 1 --model scripted

Real run:
  python runner.py --tasks tasks/dev.json --policies c0 c1 c2 --seeds 1 2 3 \
      --model openai --model-id <model> --base-url <url> --api-key env:AGENT_API_KEY
"""
from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
from datetime import datetime
from pathlib import Path

from backend import Backend
from loop import RunConfig, run_agent
from model import OpenAIChatModel, ScriptedModel
from policies import make_policy

SUMMARY_COLS = [
    "run_id", "timestamp", "runner", "task_id", "task_type", "policy", "seed",
    "max_attempts", "model_id", "success", "abstained", "model_calls",
    "tool_attempts", "repair_turns", "tokens_in", "tokens_out", "wall_seconds",
    "end_reason", "check_detail", "trace_path",
]


def resolve_api_key(value: str | None) -> str | None:
    if value is None:
        return os.environ.get("AGENT_API_KEY")
    if value.startswith("env:"):
        return os.environ.get(value[4:])
    return value


def build_model(args, task_id: str, policy: str):
    if args.model == "scripted":
        return ScriptedModel(args.script, task_id, policy)
    if args.model == "openai":
        if not args.model_id:
            raise SystemExit("--model openai requires --model-id")
        return OpenAIChatModel(args.model_id, base_url=args.base_url,
                               api_key=resolve_api_key(args.api_key))
    raise SystemExit(f"unknown --model {args.model!r}")


def main():
    ap = argparse.ArgumentParser(description="A4 minimal agent: recovery policy experiment")
    ap.add_argument("--tasks", default="tasks/dev.json")
    ap.add_argument("--policies", nargs="+", default=["c0_retry_only",
                                                      "c1_typed_feedback",
                                                      "c2_bounded_repair"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--max-attempts", type=int, default=6,
                    help="matched budget: tool executions per task")
    ap.add_argument("--max-model-calls", type=int, default=None,
                    help="default: 1 for c0, else max-attempts")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--model", choices=["scripted", "openai"], default="scripted")
    ap.add_argument("--model-id", default=None, help="e.g. glm-4.5-flash / qwen-plus")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None, help="literal key or env:VAR_NAME")
    ap.add_argument("--script", default="scripts/demo_replies.jsonl",
                    help="scripted-model trajectory file")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    with open(args.tasks, encoding="utf-8") as f:
        tasks = json.load(f)

    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.csv"
    new_file = not summary_path.exists()
    timestamp = datetime.now().isoformat(timespec="seconds")
    runner_name = getpass.getuser()

    rows = []
    for task in tasks:
        for pname in args.policies:
            policy = make_policy(pname)
            max_model_calls = args.max_model_calls
            if max_model_calls is None:
                max_model_calls = 1 if pname == "c0_retry_only" else args.max_attempts
            cfg = RunConfig(max_attempts=args.max_attempts,
                            max_model_calls=max_model_calls,
                            temperature=args.temperature)
            for seed in args.seeds:
                cfg.seed = seed
                backend = Backend(task.get("failures"))
                model = build_model(args, task["id"], pname)
                res = run_agent(task, policy, model, backend, cfg)

                trace_path = out / "traces" / task["id"] / pname / f"s{seed}.json"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=2),
                                      encoding="utf-8")

                rows.append({
                    "run_id": f"{task['id']}/{pname}/s{seed}",
                    "timestamp": timestamp, "runner": runner_name,
                    "task_id": task["id"], "task_type": task.get("type", ""),
                    "policy": pname, "seed": seed,
                    "max_attempts": cfg.max_attempts,
                    "model_id": model.model_id,
                    "success": res.success, "abstained": res.abstained,
                    "model_calls": res.stats["model_calls"],
                    "tool_attempts": res.stats["tool_attempts"],
                    "repair_turns": res.stats["repair_turns"],
                    "tokens_in": res.stats["tokens_in"],
                    "tokens_out": res.stats["tokens_out"],
                    "wall_seconds": res.stats["wall_seconds"],
                    "end_reason": res.stats["end_reason"],
                    "check_detail": res.check_detail,
                    "trace_path": str(trace_path.relative_to(out)),
                })

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        if new_file:
            w.writeheader()
        w.writerows(rows)

    # console table
    print(f"\n{len(rows)} runs -> {summary_path}")
    header = f"{'task':10} {'policy':18} {'seed':>4} {'ok':>5} {'abst':>5} {'mcall':>5} {'tcall':>5} {'repair':>6} {'wall':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['task_id']:10} {r['policy']:18} {r['seed']:>4} "
              f"{str(r['success']):>5} {str(r['abstained']):>5} "
              f"{r['model_calls']:>5} {r['tool_attempts']:>5} {r['repair_turns']:>6} "
              f"{r['wall_seconds']:>6.2f}")
    print()


if __name__ == "__main__":
    main()
