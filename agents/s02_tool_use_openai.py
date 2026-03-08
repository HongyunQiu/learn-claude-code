"""s02 (OpenAI-compatible): Tool Use via function calling.

This mirrors the lesson idea: keep the same agent loop, but add tools + handlers.
Designed for local vLLM (OpenAI-compatible) such as gpt-oss-120b.

Env vars (in .env):
  OPENAI_BASE_URL=http://172.24.168.225:8389/v1
  OPENAI_API_KEY=dummy
  OPENAI_MODEL=openai/gpt-oss-120b

Run:
  source .venv/bin/activate
  python agents/s02_tool_use_openai.py

Notes:
  - This implements a single tool: bash(command).
  - Includes a minimal safety guard against obviously destructive commands.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class ToolResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int


def _looks_dangerous(cmd: str) -> Optional[str]:
    """Best-effort guardrails. Not a sandbox."""
    s = cmd.strip().lower()

    banned_substrings = [
        "rm -rf /",
        "rm -fr /",
        "sudo ",
        "shutdown",
        "reboot",
        "mkfs",
        "dd if=",
        "diskutil erase",
        ":(){:|:&};:",  # fork bomb
    ]

    for b in banned_substrings:
        if b in s:
            return f"blocked dangerous pattern: {b}"

    # Block raw rm when targeting root or home explicitly
    if s.startswith("rm ") and (" /" in s or s.endswith(" /")):
        return "blocked rm targeting absolute path"

    return None


def tool_bash(command: str, timeout_s: int = 30) -> ToolResult:
    reason = _looks_dangerous(command)
    if reason:
        return ToolResult(False, "", reason, 126)

    # Execute in repo root to keep things contained.
    proc = subprocess.run(
        ["/bin/bash", "-lc", command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env={**os.environ},
    )
    return ToolResult(proc.returncode == 0, proc.stdout, proc.stderr, proc.returncode)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash shell command on the local machine (cwd=repo root).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run."},
                    "timeout_s": {
                        "type": "integer",
                        "description": "Timeout seconds (default 30).",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 300,
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }
]


def dispatch_tool(name: str, arguments: Dict[str, Any]) -> str:
    if name != "bash":
        return json.dumps({"ok": False, "error": f"unknown tool: {name}"}, ensure_ascii=False)

    cmd = arguments.get("command", "")
    timeout_s = int(arguments.get("timeout_s", 30))

    r = tool_bash(cmd, timeout_s=timeout_s)
    payload = {
        "ok": r.ok,
        "returncode": r.returncode,
        "stdout": r.stdout[-8000:],
        "stderr": r.stderr[-8000:],
    }
    return json.dumps(payload, ensure_ascii=False)


def run_loop(client: OpenAI, model: str) -> None:
    system = (
        "You are a coding agent. When you need to inspect files or run commands, "
        "use the provided tools (especially bash). If you use a tool, explain briefly "
        "what you are doing and then continue."
    )

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]

    print("\nType your task. This agent can call the bash tool. Ctrl+C to exit.\n")

    while True:
        user_text = input("user> ").strip()
        if not user_text:
            continue
        messages.append({"role": "user", "content": user_text})

        while True:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )

            msg = resp.choices[0].message
            # Store assistant message (may include tool_calls)
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": getattr(msg, "tool_calls", None),
                }
            )

            tool_calls = getattr(msg, "tool_calls", None)

            # --- Path A: proper function calling (tool_calls present) ---
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.function
                    name = fn.name
                    try:
                        args = json.loads(fn.arguments or "{}")
                    except Exception:
                        args = {"_raw": fn.arguments}

                    output = dispatch_tool(name, args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": output,
                        }
                    )
                # Continue inner loop (model sees tool outputs)
                continue

            # --- Path B: fallback for models that don't emit tool_calls ---
            # Some local models respond with JSON like {"command": "..."} instead.
            if msg.content:
                try:
                    obj = json.loads(msg.content)
                except Exception:
                    obj = None

                if isinstance(obj, dict) and "command" in obj:
                    output = dispatch_tool("bash", obj)
                    # Feed result back as user text (since no tool_call_id exists)
                    messages.append(
                        {
                            "role": "user",
                            "content": f"TOOL_RESULT bash\n{output}",
                        }
                    )
                    continue

                print(f"assistant> {msg.content}\n")
            else:
                print("assistant> (no content)\n")

            break


def main() -> None:
    load_dotenv(dotenv_path=os.path.join(REPO_ROOT, ".env"))

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or "dummy"
    model = os.getenv("OPENAI_MODEL")

    if not model:
        raise SystemExit("Missing env OPENAI_MODEL")

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    run_loop(client, model)


if __name__ == "__main__":
    main()
