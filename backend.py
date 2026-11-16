"""Backend: in-memory tools + deterministic failure injection + task checker.

Owner: 甲 (environment & evaluation)
Rules: keep the harness deterministic; extend tools here only with the team's
agreement after the W1 interface freeze (see docs/interface.md).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Errors and failure injection
# ---------------------------------------------------------------------------

TRANSIENT_ERRORS = ("Timeout", "RateLimit")   # retrying the same call may help
SEMANTIC_ERRORS = ("Validation",)             # retrying the same call cannot help

DEFAULT_ERROR_MESSAGES = {
    "Timeout": "tool timed out after 2000ms",
    "RateLimit": "HTTP 429: quota exceeded, retry after 10s",
    "Validation": "invalid arguments",
}


class ToolError(Exception):
    """A failure raised while executing a tool (injected or real)."""

    def __init__(self, error_type: str, message: str):
        super().__init__(f"[{error_type}] {message}")
        self.error_type = error_type
        self.message = message

    @property
    def transient(self) -> bool:
        return self.error_type in TRANSIENT_ERRORS


@dataclass
class Meeting:
    title: str
    time: str
    attendees: list = field(default_factory=list)


@dataclass
class Email:
    to: str
    subject: str
    body: str


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class Backend:
    """In-memory company backend (calendar + email outbox).

    A fresh instance is created for every run -> full isolation and replayability.

    failure_schedule: list of {"call": <logical call index>, "error": <type>,
                               "persist": bool, "message": optional}
      - "call" indexes *logical* tool calls (1-based, counted per run by the
        agent loop). Retries of the same logical call keep the same index,
        so every recovery policy faces identical failures.
      - persist=False: the error fires once (transient); later executions of
        the same logical call succeed.
      - persist=True: every execution of that logical call fails (semantic).
    """

    def __init__(self, failure_schedule: list | None = None):
        self.calendar: dict[str, Meeting] = {}
        self.outbox: list[Email] = []
        self._failures = {e["call"]: e for e in (failure_schedule or [])}
        self._fired: set[int] = set()

    # -- tool execution -----------------------------------------------------

    def execute(self, name: str, args: dict, logical_call: int) -> dict:
        self._maybe_inject(logical_call)
        if name == "book_meeting":
            return self._book_meeting(**args)
        if name == "send_email":
            return self._send_email(**args)
        raise ToolError("Validation", f"unknown tool: {name!r}")

    def _maybe_inject(self, logical_call: int) -> None:
        entry = self._failures.get(logical_call)
        if entry is None:
            return
        if not entry.get("persist", False):
            if logical_call in self._fired:
                return
            self._fired.add(logical_call)
        raise ToolError(
            entry["error"],
            entry.get("message", DEFAULT_ERROR_MESSAGES.get(entry["error"], "tool failed")),
        )

    # -- tools --------------------------------------------------------------

    def _book_meeting(self, title: str, time: str, attendees: list) -> dict:
        if not isinstance(title, str) or not title:
            raise ToolError("Validation", f"title must be a non-empty string, got {title!r}")
        if not isinstance(time, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", time):
            raise ToolError("Validation", f"time must be 'YYYY-MM-DD HH:MM', got {time!r}")
        if not attendees or not all(isinstance(a, str) and a for a in attendees):
            raise ToolError("Validation", "attendees must be a non-empty list of names")
        for m in self.calendar.values():
            if m.time == time and set(m.attendees) & set(attendees):
                raise ToolError(
                    "Validation",
                    f"conflict with meeting {m.title!r} at {time} "
                    f"for {sorted(set(m.attendees) & set(attendees))}",
                )
        meeting = Meeting(title, time, list(attendees))
        self.calendar[title] = meeting
        return {"status": "booked", "title": meeting.title, "time": meeting.time,
                "attendees": meeting.attendees}

    def _send_email(self, to: str, subject: str, body: str) -> dict:
        if not isinstance(to, str) or "@" not in to:
            raise ToolError("Validation", f"'to' must contain '@', got {to!r}")
        if not subject:
            raise ToolError("Validation", "subject must be a non-empty string")
        if not body:
            raise ToolError("Validation", "body must be a non-empty string")
        email = Email(to, subject, body)
        self.outbox.append(email)
        return {"status": "sent", "to": email.to, "subject": email.subject}


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "book_meeting",
            "description": "Book a meeting in the company calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Meeting title"},
                    "time": {"type": "string",
                             "description": "Meeting time, format 'YYYY-MM-DD HH:MM'"},
                    "attendees": {"type": "array", "items": {"type": "string"},
                                  "description": "Attendee names"},
                },
                "required": ["title", "time", "attendees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Task checker: executable ground truth, never the model itself
# ---------------------------------------------------------------------------

def check_task(backend: Backend, no_tool_calls: bool, expected: dict) -> tuple[bool, str]:
    """Evaluate a task's expected block against the final backend state.

    expected schema:
      {
        "no_tool_calls": true | false,          # for ambiguous tasks
        "calendar": {"empty": true} | {"contains_meeting": [{"title", "time", "attendees"}]},
        "outbox":   {"empty": true} | {"contains_email":   [{"to", "subject", "body_contains"}]}
      }
    Returns (ok, detail_string).
    """
    ok = True
    checks: list[str] = []

    if "no_tool_calls" in expected:
        want = bool(expected["no_tool_calls"])
        ok &= no_tool_calls == want
        checks.append(f"no_tool_calls={no_tool_calls} (want {want})")

    cal = expected.get("calendar", {})
    if cal.get("empty") and backend.calendar:
        ok = False
        checks.append("calendar should be empty")
    for want in cal.get("contains_meeting", []):
        m = backend.calendar.get(want["title"])
        match = (m is not None and m.time == want["time"]
                 and set(m.attendees) == set(want["attendees"]))
        ok &= match
        checks.append(f"meeting {want['title']!r}: {'ok' if match else 'missing/mismatch'}")

    ob = expected.get("outbox", {})
    if ob.get("empty") and backend.outbox:
        ok = False
        checks.append("outbox should be empty")
    for want in ob.get("contains_email", []):
        hit = any(e.to == want["to"] and e.subject == want["subject"]
                  and want.get("body_contains", "") in e.body
                  for e in backend.outbox)
        ok &= hit
        checks.append(f"email to {want['to']!r}: {'ok' if hit else 'missing/mismatch'}")

    return bool(ok), "; ".join(checks) if checks else "no assertions"
