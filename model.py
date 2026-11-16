"""Chat-model wrappers: real API model + scripted model for offline smoke tests.

Owner: 乙 (agent loop); 甲 curates the script files under scripts/.
Interface is frozen: ChatModel.reply(messages, tools, temperature, seed) -> AssistantReply.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class AssistantReply:
    content: str | None
    tool_calls: list = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0


class ChatModel(ABC):
    model_id: str = "base"

    @abstractmethod
    def reply(self, messages: list, tools: list, temperature: float,
              seed: int | None) -> AssistantReply:
        """One chat-completion call.

        messages: OpenAI chat format (system/user/assistant/tool roles).
        tools:    OpenAI tool schemas (from backend.TOOL_SCHEMAS).
        Must return either content (final answer) or one or more tool calls.
        """


# ---------------------------------------------------------------------------
# Scripted model: replays a fixed trajectory, for harness smoke tests only.
# Never use it to measure policy quality.
# ---------------------------------------------------------------------------

class ScriptedModel(ChatModel):
    """Replays replies for a task from a JSONL script.

    JSONL format, one line per task:
      {"task_id": "dev_001", "policy": null, "replies": [
          {"content": null, "tool_calls": [{"name": "book_meeting", "args": {...}}]},
          {"content": "Meeting booked.", "tool_calls": []}
      ]}
    A line with "policy": "c2" overrides the default sequence for that policy.
    When the script is exhausted, replies with a fixed final message.
    """

    def __init__(self, path: str, task_id: str, policy: str | None = None):
        self.model_id = f"scripted:{task_id}"
        scripts: dict[tuple[str, str | None], list] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                scripts[(e["task_id"], e.get("policy"))] = e.get("replies", [])
        self._replies = list(scripts.get((task_id, policy)) or scripts.get((task_id, None)) or [])
        self._idx = 0

    def reply(self, messages, tools, temperature, seed):
        if self._idx < len(self._replies):
            r = self._replies[self._idx]
            self._idx += 1
            calls = [ToolCall(c["name"], c["args"]) for c in r.get("tool_calls", [])]
            content = r.get("content")
            # rough token estimate so cost columns stay meaningful in smoke runs
            n_in = sum(len(str(m.get("content") or "")) // 4 for m in messages)
            n_out = len(content or "") // 4 + sum(len(json.dumps(c.args)) // 4 for c in calls)
            return AssistantReply(content=content, tool_calls=calls,
                                  tokens_in=n_in, tokens_out=n_out)
        return AssistantReply(content="OK, done.", tool_calls=[], tokens_in=0, tokens_out=0)


# ---------------------------------------------------------------------------
# Real API model (OpenAI-compatible; works with GLM / Qwen / DeepSeek etc.)
# ---------------------------------------------------------------------------

class OpenAIChatModel(ChatModel):
    def __init__(self, model_id: str, base_url: str | None = None,
                 api_key: str | None = None):
        import openai  # lazy import: offline smoke tests do not need it
        self.model_id = model_id
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def reply(self, messages, tools, temperature, seed):
        kwargs = dict(model=self.model_id, messages=messages, tools=tools,
                      temperature=temperature)
        if seed is not None:
            kwargs["seed"] = seed  # remove this line if your provider rejects it
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = []
        for t in (msg.tool_calls or []):
            try:
                args = json.loads(t.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(t.function.name, args))
        usage = resp.usage
        return AssistantReply(
            content=msg.content,
            tool_calls=calls,
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
        )
