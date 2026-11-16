"""Minimal agent loop: one chat-completion call + tools + our own control code.

No agent framework. Owner: 乙 (agent loop).
Budget semantics (frozen, see docs/interface.md):
  - max_attempts: shared matched budget = tool EXECUTIONS per task (all arms).
  - max_model_calls: model consultations; runner sets 1 for c0, max_attempts
    for c1/c2. Cost metric = model calls / tokens, reported per run.
"""
from __future__ import annotations

import dataclasses
import json
import random
import time
from dataclasses import dataclass, field

from backend import TOOL_SCHEMAS, ToolError, check_task
from policies import FailureInfo, RecoveryPolicy


SYSTEM_PROMPT = (
    "You are a company assistant. You can call two tools: book_meeting and "
    "send_email. Only call a tool when the request contains all needed "
    "information; if anything is missing or ambiguous, ask the user instead "
    "of guessing. Answer briefly."
)


@dataclass
class RunConfig:
    max_attempts: int = 6      # matched budget: tool executions per task
    max_model_calls: int = 6   # c0 runs with 1
    temperature: float = 0.7
    seed: int = 1


@dataclass
class RunResult:
    task_id: str
    policy: str
    seed: int
    model_id: str
    config: RunConfig
    events: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    success: bool = False
    abstained: bool = False
    final_content: str = ""
    check_detail: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "policy": self.policy,
            "seed": self.seed,
            "model_id": self.model_id,
            "config": dataclasses.asdict(self.config),
            "stats": self.stats,
            "events": self.events,
            "success": self.success,
            "abstained": self.abstained,
            "final_content": self.final_content,
            "check_detail": self.check_detail,
        }


def run_agent(task: dict, policy: RecoveryPolicy, model, backend,
              cfg: RunConfig) -> RunResult:
    """Run one task under one recovery policy with a fresh backend."""
    random.seed(cfg.seed)  # the harness itself stays deterministic

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["user"]},
    ]
    events: list[dict] = []
    model_calls = tool_attempts = repair_turns = logical = 0
    tokens_in = tokens_out = 0
    final_content = ""
    ended = None
    t0 = time.time()

    while model_calls < cfg.max_model_calls and tool_attempts < cfg.max_attempts:
        reply = model.reply(messages, TOOL_SCHEMAS, cfg.temperature, cfg.seed)
        model_calls += 1
        tokens_in += reply.tokens_in
        tokens_out += reply.tokens_out
        events.append({
            "t": "model", "n": model_calls, "content": reply.content,
            "tool_calls": [{"name": c.name, "args": c.args} for c in reply.tool_calls],
        })

        if not reply.tool_calls:                      # final answer
            final_content = reply.content or ""
            ended = "final_answer"
            break

        base = logical                                # stable ids for this reply
        ids = [f"call_{base + 1 + k}" for k in range(len(reply.tool_calls))]
        messages.append({
            "role": "assistant", "content": reply.content,
            "tool_calls": [
                {"id": ids[k], "type": "function", "function": {
                    "name": c.name, "arguments": json.dumps(c.args, ensure_ascii=False)}}
                for k, c in enumerate(reply.tool_calls)
            ],
        })

        for i, tc in enumerate(reply.tool_calls):
            logical += 1
            tool_id = ids[i]
            attempts_on_call = 0
            while True:
                if tool_attempts >= cfg.max_attempts:
                    events.append({"t": "budget_exhausted", "logical_call": logical})
                    messages.append({"role": "tool", "tool_call_id": tool_id,
                                     "content": json.dumps({"note": "tool attempt budget exhausted"})})
                    ended = "attempt_budget"
                    break
                attempts_on_call += 1
                tool_attempts += 1
                try:
                    result = backend.execute(tc.name, tc.args, logical)
                    events.append({
                        "t": "tool", "logical_call": logical, "attempt": attempts_on_call,
                        "name": tc.name, "args": tc.args, "ok": True, "result": result,
                    })
                    messages.append({"role": "tool", "tool_call_id": tool_id,
                                     "content": json.dumps(result, ensure_ascii=False)})
                    break
                except ToolError as e:
                    events.append({
                        "t": "failure", "logical_call": logical, "attempt": attempts_on_call,
                        "error_type": e.error_type, "message": e.message,
                        "transient": e.transient,
                    })
                    action = policy.on_failure(FailureInfo(
                        logical_call=logical, error_type=e.error_type, message=e.message,
                        transient=e.transient, attempts_on_call=attempts_on_call,
                        repair_turns_used=repair_turns,
                        attempts_budget_left=cfg.max_attempts - tool_attempts,
                    ))
                    events.append({"t": "policy", "logical_call": logical,
                                   "action": action.kind})
                    if action.kind == "retry":
                        if action.backoff_seconds:
                            time.sleep(action.backoff_seconds)
                        continue
                    if action.kind == "replan":
                        repair_turns += 1
                        messages.append({"role": "tool", "tool_call_id": tool_id,
                                         "content": json.dumps(
                                             {"error_type": e.error_type, "message": e.message},
                                             ensure_ascii=False)})
                    else:  # give_up
                        messages.append({"role": "tool", "tool_call_id": tool_id,
                                         "content": json.dumps(
                                             {"error_type": e.error_type, "message": e.message,
                                              "note": "recovery budget exhausted"},
                                             ensure_ascii=False)})
                    break
            if ended:
                break
        if ended:
            break

    no_tool_calls = tool_attempts == 0
    ok, detail = check_task(backend, no_tool_calls, task.get("expected", {}))
    return RunResult(
        task_id=task["id"], policy=policy.name, seed=cfg.seed,
        model_id=model.model_id, config=cfg, events=events,
        stats={
            "model_calls": model_calls, "tool_attempts": tool_attempts,
            "repair_turns": repair_turns, "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "wall_seconds": round(time.time() - t0, 2),
            "end_reason": ended,
        },
        success=ok, abstained=(no_tool_calls and bool(final_content)),
        final_content=final_content, check_detail=detail,
    )
