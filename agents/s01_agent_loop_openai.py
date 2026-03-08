"""s01 variant: run the minimal agent loop using an OpenAI-compatible API.

This is a pragmatic adapter so you can run the lessons with a local vLLM model
(e.g. gpt-oss-120b) that speaks the OpenAI API, not Anthropic.

Env vars:
  OPENAI_BASE_URL (required for local vLLM, e.g. http://172.24.168.225:8389/v1)
  OPENAI_API_KEY  (can be dummy for local vLLM)
  OPENAI_MODEL    (e.g. openai/gpt-oss-120b)

Usage:
  python agents/s01_agent_loop_openai.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI


def main():
    load_dotenv()

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")

    if not model:
        raise SystemExit("Missing env OPENAI_MODEL")

    client = OpenAI(
        api_key=api_key or "dummy",
        base_url=base_url or None,
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    print("\nType your message. Ctrl+C to exit.\n")

    while True:
        user_text = input("user> ").strip()
        if not user_text:
            continue
        messages.append({"role": "user", "content": user_text})

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )

        assistant_text = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_text})

        print(f"assistant> {assistant_text}\n")


if __name__ == "__main__":
    main()
