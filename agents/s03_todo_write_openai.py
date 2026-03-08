"""s03 (OpenAI-compatible): TodoWrite

Adds a structured todo tool that the model uses to track progress.
Also injects a nag reminder if the model forgets to update todos.

This file is designed for local vLLM models that speak the OpenAI Chat
Completions API (e.g. gpt-oss-120b).

Env vars (in .env):
  OPENAI_BASE_URL=http://172.24.168.225:8389/v1
  OPENAI_API_KEY=dummy
  OPENAI_MODEL=openai/gpt-oss-120b

Run:
  source .venv/bin/activate
  python agents/s03_todo_write_openai.py

Notes about tool calling:
  - If the model emits proper tool_calls, we execute them.
  - If it does not, we fallback to a JSON protocol. Supported forms:
      {"tool": "bash", "args": {"command": "ls"}}
      {"tool": "todo", "args": {"items": [...]}}
      {"command": "ls"}   # treated as bash
    This keeps the lesson runnable on models that don't support tool_calls.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKDIR = Path.cwd()


# ---------------- Todo Manager ----------------
class TodoManager:
    def __init__(self):
        self.items: List[Dict[str, str]] = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


# ---------------- Tooling ----------------
@dataclass
class ToolResult:
    ok: bool
    payload: Dict[str, Any]


def _looks_dangerous(cmd: str) -> Optional[str]:
    s = cmd.strip().lower()
    banned = [
        "rm -rf /",
        "rm -fr /",
        "sudo ",
        "shutdown",
        "reboot",
        "mkfs",
        "dd if=",
        "diskutil erase",
        ":(){:|:&};:",
    ]
    for b in banned:
        if b in s:
            return f"blocked dangerous pattern: {b}"
    if s.startswith("rm ") and (" /" in s or s.endswith(" /")):
        return "blocked rm targeting absolute path"
    return None


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    try:
        if not path.is_relative_to(WORKDIR):
            raise ValueError
    except AttributeError:
        # py<3.9 fallback (not needed here, but kept for portability)
        if not str(path).startswith(str(WORKDIR.resolve())):
            raise ValueError
    return path


def tool_bash(command: str, timeout_s: int = 120) -> ToolResult:
    reason = _looks_dangerous(command)
    if reason:
        return ToolResult(False, {"error": reason, "returncode": 126})
    try:
        r = subprocess.run(
            ["/bin/bash", "-lc", command],
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ},
        )
        return ToolResult(
            r.returncode == 0,
            {
                "returncode": r.returncode,
                "stdout": (r.stdout or "")[-8000:],
                "stderr": (r.stderr or "")[-8000:],
            },
        )
    except subprocess.TimeoutExpired:
        return ToolResult(False, {"error": "timeout", "returncode": 124})


def tool_read_file(path: str, limit: Optional[int] = None) -> ToolResult:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return ToolResult(True, {"content": "\n".join(lines)[:8000]})
    except Exception as e:
        return ToolResult(False, {"error": str(e)})


def tool_write_file(path: str, content: str) -> ToolResult:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return ToolResult(True, {"written": len(content)})
    except Exception as e:
        return ToolResult(False, {"error": str(e)})


def tool_edit_file(path: str, old_text: str, new_text: str) -> ToolResult:
    try:
        fp = safe_path(path)
        src = fp.read_text()
        if old_text not in src:
            return ToolResult(False, {"error": "text_not_found"})
        fp.write_text(src.replace(old_text, new_text, 1))
        return ToolResult(True, {"edited": str(path)})
    except Exception as e:
        return ToolResult(False, {"error": str(e)})


def tool_todo(items: list) -> ToolResult:
    try:
        rendered = TODO.update(items)
        return ToolResult(True, {"todos": rendered})
    except Exception as e:
        return ToolResult(False, {"error": str(e)})


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command (cwd=current working dir).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_s": {"type": "integer", "default": 120, "minimum": 1, "maximum": 300},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo",
            "description": "Update the todo list to track progress.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                            },
                            "required": ["id", "text", "status"],
                        },
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            },
        },
    },
]


def dispatch(tool: str, args: Dict[str, Any]) -> str:
    if tool == "bash":
        r = tool_bash(args.get("command", ""), int(args.get("timeout_s", 120)))
    elif tool == "read_file":
        r = tool_read_file(args.get("path", ""), args.get("limit"))
    elif tool == "write_file":
        r = tool_write_file(args.get("path", ""), args.get("content", ""))
    elif tool == "edit_file":
        r = tool_edit_file(args.get("path", ""), args.get("old_text", ""), args.get("new_text", ""))
    elif tool == "todo":
        r = tool_todo(args.get("items", []))
    else:
        r = ToolResult(False, {"error": f"unknown_tool:{tool}"})
    return json.dumps({"ok": r.ok, **r.payload}, ensure_ascii=False)


# ---------------- Agent Loop ----------------
def _extract_tool_request_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Fallback protocol.

    Supported:
      - {"tool": "bash", "args": {...}}
      - {"command": "..."}  (treated as bash)
      - {"path": "...", "limit": 200} (treated as read_file)
    """
    try:
        obj = json.loads(text)
    except Exception:
        return None

    if isinstance(obj, dict) and isinstance(obj.get("tool"), str) and isinstance(obj.get("args"), dict):
        return obj

    # Common local-model patterns without explicit tool name
    # 1) {"command": "..."} -> bash
    if isinstance(obj, dict) and isinstance(obj.get("command"), str):
        return {"tool": "bash", "args": {"command": obj["command"]}}

    # 2) {"path": "...", "limit": N} -> read_file
    if isinstance(obj, dict) and isinstance(obj.get("path"), str):
        args = {"path": obj["path"]}
        if isinstance(obj.get("limit"), int):
            args["limit"] = obj["limit"]
        return {"tool": "read_file", "args": args}

    return None


def agent_loop(client: OpenAI, model: str, messages: List[Dict[str, Any]]) -> None:
    rounds_since_todo = 0
    system = (
        f"You are a coding agent at {WORKDIR}. "
        "You MUST maintain a todo list for any multi-step task using the todo tool. "
        "Mark one item in_progress while working, and completed when done. "
        "Prefer tools over prose when you need to inspect files or run commands.\n\n"
        "Tool calling:\n"
        "- If function calling is available, call tools normally.\n"
        "- Otherwise, when you want a tool, respond ONLY with JSON in one of these forms:\n"
        "    {\"tool\": <name>, \"args\": {...}}\n"
        "    {\"command\": \"...\"}   (alias for bash)\n"
        "    {\"path\": \"...\", \"limit\": 200} (alias for read_file)"
    )
    messages.insert(0, {"role": "system", "content": system})

    while True:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message

        # store assistant message
        tool_calls = getattr(msg, "tool_calls", None)
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": tool_calls,
            }
        )

        used_todo = False

        # Path A: proper tool_calls
        if tool_calls:
            tool_results = []
            for tc in tool_calls:
                fn = tc.function
                name = fn.name
                try:
                    args = json.loads(fn.arguments or "{}")
                except Exception:
                    args = {}
                if name == "todo":
                    used_todo = True
                out = dispatch(name, args)
                tool_results.append((tc.id, out))

            # Update nag counter
            rounds_since_todo = 0 if used_todo else rounds_since_todo + 1

            # Feed tool outputs back
            for tool_call_id, out in tool_results:
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": out})

            if rounds_since_todo >= 3:
                messages.append({"role": "user", "content": "<reminder>Update your todos.</reminder>"})

            continue

        # Path B: fallback JSON tool protocol
        if msg.content:
            req = _extract_tool_request_from_text(msg.content)
            if req:
                if req["tool"] == "todo":
                    used_todo = True
                out = dispatch(req["tool"], req["args"])
                rounds_since_todo = 0 if used_todo else rounds_since_todo + 1

                # Without tool_call_id, feed as user text
                tool_blob = f"TOOL_RESULT {req['tool']}\n{out}"
                if rounds_since_todo >= 3:
                    tool_blob = "<reminder>Update your todos.</reminder>\n" + tool_blob
                messages.append({"role": "user", "content": tool_blob})
                continue

            # No tool request; print and exit this turn
            return

        return


def main() -> None:
    load_dotenv(dotenv_path=str(REPO_ROOT / ".env"))
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or "dummy"
    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise SystemExit("Missing env OPENAI_MODEL")

    client = OpenAI(api_key=api_key, base_url=base_url or None)

    history: List[Dict[str, Any]] = []
    print("\nType your task. This agent has tools + todo tracking. Ctrl+C to exit.\n")
    while True:
        try:
            q = input("s03> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "exit") or not q:
            break
        history.append({"role": "user", "content": q})
        # Each turn: run until the agent returns normal text
        agent_loop(client, model, history)
        # Print the last assistant text response (if any)
        for m in reversed(history):
            if m["role"] == "assistant" and m.get("content"):
                print(f"assistant> {m['content']}\n")
                break


if __name__ == "__main__":
    main()
