"""Pack polished answers as Qwen chat-templated SFT rows (`text` column)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import POLISHED_PATH, SFT_PATH

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def apply_chat_template(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    tokenize: bool = False,
) -> str:
    """Qwen chat template (same turn markers as tokenizer.apply_chat_template)."""
    if tokenize:
        raise ValueError("this helper returns a string; tokenize=False only")
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def pack_sft_row(record: dict[str, Any]) -> dict[str, Any]:
    instruction = record.get("instruction") or "다음 조건에 맞는 한국어 글을 작성하라."
    answer = record["answer"]
    messages = [
        {
            "role": "system",
            "content": (
                "당신은 원어민 한국어 화자처럼 글을 쓴다. "
                "문체와 환경을 지키고, 번역투와 AI 티를 넣지 않는다."
            ),
        },
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": answer},
    ]
    text = apply_chat_template(messages)
    return {
        "id": record["id"],
        "text": text,
        "instruction": instruction,
        "answer": answer,
        "topic": record["topic"],
        "environment": record["environment"],
        "register": record["register"],
        "age": record["age"],
        "background": record["background"],
    }


def pack_corpus(src: Path = POLISHED_PATH, dst: Path = SFT_PATH) -> Path:
    from .generate import load_jsonl

    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = [pack_sft_row(rec) for rec in load_jsonl(src)]
    with dst.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return dst


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Pack Qwen-templated SFT JSONL")
    p.add_argument("--src", type=Path, default=POLISHED_PATH)
    p.add_argument("--dst", type=Path, default=SFT_PATH)
    args = p.parse_args(argv)
    pack_corpus(src=args.src, dst=args.dst)
    print(f"packed → {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
