"""s04 (OpenAI-compatible): Subagents

Spawn a child agent with fresh messages=[]. The child shares the filesystem,
uses the same tools, then returns only a summary to the parent.

Env vars (in .env):
  OPENAI_BASE_URL=http://172.24.168.225:8389/v1
  OPENAI_API_KEY=dummy
  OPENAI_MODEL=openai/gpt-oss-120b

Run:
  source .venv/bin/activate
  python agents/s04_subagent_openai.py

Tool calling support:
  - Uses proper tool_calls when available.
  - Fallback JSON protocol when not:
      {"tool":"bash","args":{...}}
      {"tool":"task","args":{"prompt":"...","description":"..."}}
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


# ---------------- Base tools shared by parent/child ----------------
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


BASE_TOOLS = [
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
]


def dispatch_base(tool: str, args: Dict[str, Any]) -> str:
    if tool == "bash":
        r = tool_bash(args.get("command", ""), int(args.get("timeout_s", 120)))
    elif tool == "read_file":
        r = tool_read_file(args.get("path", ""), args.get("limit"))
    elif tool == "write_file":
        r = tool_write_file(args.get("path", ""), args.get("content", ""))
    elif tool == "edit_file":
        r = tool_edit_file(args.get("path", ""), args.get("old_text", ""), args.get("new_text", ""))
    else:
        r = ToolResult(False, {"error": f"unknown_tool:{tool}"})
    return json.dumps({"ok": r.ok, **r.payload}, ensure_ascii=False)


# ---------------- Subagent ----------------
def _extract_tool_request_from_text(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
    except Exception:
        return None

    if isinstance(obj, dict) and isinstance(obj.get("tool"), str) and isinstance(obj.get("args"), dict):
        return obj

    # Common local-model pattern: {"command": "..."}
    if isinstance(obj, dict) and isinstance(obj.get("command"), str):
        return {"tool": "bash", "args": {"command": obj["command"]}}

    return None


def run_child_agent(client: OpenAI, model: str, prompt: str) -> str:
    sub_system = (
        f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize findings. "
        "You may use tools to inspect the filesystem.\n\n"
        "Tool calling:\n"
        "- If function calling is available, call tools normally.\n"
        "- Otherwise, when you want a tool, respond ONLY with JSON: {\"tool\": <name>, \"args\": {...}}"
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": sub_system},
        {"role": "user", "content": prompt},
    ]

    for _ in range(30):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=BASE_TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": tool_calls})

        if tool_calls:
            for tc in tool_calls:
                fn = tc.function
                try:
                    args = json.loads(fn.arguments or "{}")
                except Exception:
                    args = {}
                out = dispatch_base(fn.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            continue

        if msg.content:
            req = _extract_tool_request_from_text(msg.content)
            if req:
                out = dispatch_base(req["tool"], req["args"])
                messages.append({"role": "user", "content": f"TOOL_RESULT {req['tool']}\n{out}"})
                continue

            # normal final text -> return it
            return msg.content

        return "(no output)"

    return "(child reached safety limit)"


# ---------------- Parent agent ----------------
PARENT_TOOLS = BASE_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Spawn a subagent with fresh context. It shares filesystem but not conversation history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
    }
]


def dispatch_parent(client: OpenAI, model: str, tool: str, args: Dict[str, Any]) -> str:
    if tool == "task":
        prompt = args.get("prompt", "")
        summary = run_child_agent(client, model, prompt)
        return json.dumps({"ok": True, "summary": summary}, ensure_ascii=False)
    return dispatch_base(tool, args)


def parent_loop(client: OpenAI, model: str, messages: List[Dict[str, Any]]) -> None:
    system = (
        f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks.\n\n"
        "Tool calling:\n"
        "- If function calling is available, call tools normally.\n"
        "- Otherwise, when you want a tool, respond ONLY with JSON: {\"tool\": <name>, \"args\": {...}}"
    )
    messages.insert(0, {"role": "system", "content": system})

    while True:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=PARENT_TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": tool_calls})

        if tool_calls:
            for tc in tool_calls:
                fn = tc.function
                try:
                    args = json.loads(fn.arguments or "{}")
                except Exception:
                    args = {}
                out = dispatch_parent(client, model, fn.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            continue

        if msg.content:
            req = _extract_tool_request_from_text(msg.content)
            if req:
                out = dispatch_parent(client, model, req["tool"], req["args"])
                messages.append({"role": "user", "content": f"TOOL_RESULT {req['tool']}\n{out}"})
                continue

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
    print("\nType your task. This agent can spawn subagents via task tool. Ctrl+C to exit.\n")
    while True:
        try:
            q = input("s04> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "exit") or not q:
            break
        history.append({"role": "user", "content": q})
        parent_loop(client, model, history)
        for m in reversed(history):
            if m["role"] == "assistant" and m.get("content"):
                print(f"assistant> {m['content']}\n")
                break


if __name__ == "__main__":
    main()
