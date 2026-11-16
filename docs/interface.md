# 接口冻结文档（Interface Freeze）

> 2026-11-15 起草 · **第 1 周末（11/20 前）三人共同确认后冻结**
> 冻结后任何修改都需三人在群里确认。每人只改自己负责的文件。

## 0. 文件归属

| 文件 | 负责人 | 内容 |
|---|---|---|
| `backend.py` | 甲 | 假后端 + 故障注入 + `TOOL_SCHEMAS` + `check_task` |
| `tasks/` | 甲 | dev / heldout 任务集 |
| `loop.py` | 乙 | 最小 agent 循环、`RunConfig`、`RunResult`、轨迹事件 |
| `model.py` | 乙 | `ChatModel` 接口 + OpenAI 实现 + ScriptedModel |
| `scripts/` | 甲 | 冒烟测试用的脚本化模型轨迹 |
| `policies.py` | 丙 | 三个恢复策略（C0/C1/C2） |
| `runner.py` | 丙 | 实验运行器 → traces + `results/summary.csv` |
| `docs/interface.md` | 三人 | 本文件 |

## 1. 任务 JSON schema（`tasks/*.json`，一个 JSON 数组）

```json
{
  "id": "dev_001",
  "type": "single | chained | ambiguous",
  "user": "自然语言请求",
  "failures": [{"call": 1, "error": "RateLimit", "persist": false}],
  "expected": {
    "no_tool_calls": true,                 // 仅歧义任务需要
    "calendar": {"empty": true} | {"contains_meeting": [
        {"title": "...", "time": "YYYY-MM-DD HH:MM", "attendees": ["..."]}
    ]},
    "outbox": {"empty": true} | {"contains_email": [
        {"to": "...", "subject": "...", "body_contains": "..."}
    ]}
  }
}
```

- `failures[].call` = **逻辑调用序号**（1 起）。重试同一调用不增加序号，所以三个策略面对的故障完全一致。
- `persist: false` = 瞬态（只注入一次，重试可成功）；`persist: true` = 持久（该逻辑调用的每次执行都失败）。
- `error` ∈ {Timeout, RateLimit, Validation}。
- **heldout.json 一旦冻结，任何人不得因实验结果修改。**

## 2. 策略接口（`policies.py`）

```python
FailureInfo(logical_call, error_type, message, transient,
            attempts_on_call, repair_turns_used, attempts_budget_left)
RecoveryAction(kind)   # kind ∈ {"retry", "replan", "give_up"}
```

**铁律**：策略只做决定——不调用模型、不碰后端、不读 ground truth。执行与计数全部在 `loop.py`。

## 3. 模型接口（`model.py`）

```python
reply(messages, tools, temperature, seed) -> AssistantReply(content, tool_calls, tokens_in, tokens_out)
```

`messages` 为 OpenAI 格式（system/user/assistant/tool）。真实模型用 `OpenAIChatModel`（GLM/Qwen/DeepSeek 均兼容），冒烟测试用 `ScriptedModel`（**只能测 harness，不能测策略好坏**）。

## 4. 预算语义（匹配预算的核心）

- `max_attempts`：**工具执行次数**上限（三个条件共享，默认 6）。
- `max_model_calls`：模型调用上限。**C0 固定为 1**（朴素重试从不咨询模型）；C1/C2 默认 = max_attempts。
- 成本指标 = 模型调用数 / token；墙钟时间单独报告。
- 每个运行：新 `Backend` 实例、新 `ScriptedModel` 实例 → 完全隔离。

## 5. 轨迹事件格式（`RunResult.events`，可重放）

| `t` | 含义 | 关键字段 |
|---|---|---|
| `model` | 一次模型调用 | n, content, tool_calls |
| `tool` | 一次成功的工具执行 | logical_call, attempt, name, args, result |
| `failure` | 一次失败 | logical_call, attempt, error_type, message, transient |
| `policy` | 策略决定 | logical_call, action |
| `budget_exhausted` | 预算耗尽 | logical_call |

## 6. `results/summary.csv` 列（实验日志）

run_id, timestamp, runner, task_id, task_type, policy, seed, max_attempts, model_id, success, abstained, model_calls, tool_attempts, repair_turns, tokens_in, tokens_out, wall_seconds, end_reason, check_detail, trace_path

**每跑一次实验必须用 runner 跑并追加进 summary.csv**——这是报告数据表和贡献声明的唯一证据来源。

## 7. 协作规则

1. 只改自己负责的文件；本文件与任务 schema 改动需三人确认。
2. 每天 push；commit 信息写清楚做了什么。
3. 每个节点的"完成"要有可执行证据：命令 + 输出。
4. API key 只由丙管理，不提交到仓库（`.gitignore` 已排除 `.env`）。
