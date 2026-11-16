# A4: A Minimal Agent with a Recovery Mechanism

CSC5051 NLP Final Project (Fall 2026) · Track A, brief A4 · 3-person team · API only, no GPU

**Question:** when a tool call fails, which recovery policy maximizes success per unit
cost at a matched attempt budget — plain retry (C0), typed error feedback (C1), or a
bounded repair policy (C2)?

- Plan: `A4_项目计划.md` · Proposal: `proposal_A4.md` · Interfaces: `docs/interface.md`

## Layout

```
backend.py     # 甲: in-memory tools + deterministic failure injection + task checker
loop.py        # 乙: minimal agent loop (no framework), budgets, trace events
model.py       # 乙: ChatModel interface; OpenAI-compatible client; ScriptedModel
policies.py    # 丙: C0 retry-only / C1 typed feedback / C2 bounded repair
runner.py      # 丙: tasks x policies x seeds -> traces + results/summary.csv
tasks/         # 甲: dev.json (tunable) + heldout.json (frozen in W1, do not touch)
scripts/       # 甲: scripted trajectories for offline smoke tests
docs/          # interface freeze
```

## Quick start (smoke test, no API key needed)

```
python runner.py --tasks tasks/dev.json --policies c0_retry_only c1_typed_feedback c2_bounded_repair --seeds 1 --model scripted
```

Expected smoke pattern (demonstrates the mechanism, not real results):
- `dev_001` (transient RateLimit): all three policies pass.
- `dev_002` (persistent Validation): **c0 fails** (retrying the same bad args cannot help),
  c1/c2 pass after re-planning with corrected arguments.
- `dev_003` (chained, no failure): **c0 fails** (single model call cannot finish the chain),
  c1/c2 pass.
- `dev_004` (chained, RateLimit on the 2nd call): c0 fails; c1/c2 pass.
- `dev_005` (ambiguous): all policies abstain and pass.
- `dev_006` (transient Timeout): all three policies pass.

## Real runs (OpenAI-compatible API: GLM / Qwen / DeepSeek)

```
pip install -r requirements.txt
set AGENT_API_KEY=<your key>
python runner.py --tasks tasks/dev.json ^
    --policies c0_retry_only c1_typed_feedback c2_bounded_repair ^
    --seeds 1 2 3 --max-attempts 6 ^
    --model openai --model-id <model> --base-url <provider base url> ^
    --api-key env:AGENT_API_KEY
```

(On POSIX shells replace `^` with `\`.) If your provider rejects the `seed` parameter,
remove the `kwargs["seed"] = seed` line in `model.py`.

## Outputs

- `results/summary.csv` — one row per run (the team experiment log; append-only).
- `results/traces/<task_id>/<policy>/s<seed>.json` — full replayable trace:
  every model call, tool execution, injected failure, policy decision, budgets.

## Rules

- Held-out split (`tasks/heldout.json`) is frozen in week 1. Never tune prompts on it.
- The checker is executable state checks — never let the model judge itself.
- Record model id + provider + date for every experiment (column `model_id` + log).
- ScriptedModel trajectories are for harness smoke tests only; never use them to
  measure policy quality.
