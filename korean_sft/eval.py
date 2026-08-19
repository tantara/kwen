"""Eval harness: score natural Korean, optionally generate with Unsloth.

CPU:
    python -m korean_sft eval --score-only
    python -m korean_sft eval --predictions path.jsonl

GPU (base vs LoRA):
    python -m korean_sft eval --model Qwen/Qwen3.5-4B \\
        --adapter outputs/qwen35-4b-onepage --out reports/eval-4b.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from .eval_metrics import scenario_track, score_text, summarize_run
from .pack import IM_END, IM_START, apply_chat_template
from .paths import EVAL_SCENARIOS

SYSTEM = (
    "당신은 원어민 한국어 화자처럼 글을 쓴다. "
    "문체와 환경을 지키고, 번역투와 AI 티를 넣지 않는다."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_for(scenario: dict[str, Any]) -> str:
    return apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": scenario["instruction"]},
        ],
        add_generation_prompt=True,
    )


def score_prediction(scenario: dict[str, Any], text: str) -> dict[str, Any]:
    metrics = score_text(text, scenario)
    return {
        "id": scenario.get("id"),
        "topic": scenario.get("topic"),
        "register": scenario.get("register"),
        "speech_level": scenario.get("speech_level"),
        "relation": scenario.get("relation"),
        "generation_axis": scenario.get("generation"),
        "track": metrics.get("track") or scenario_track(scenario),
        "output": text,
        **metrics,
    }


def _load_unsloth(model_name: str, adapter: str | None, max_seq: int):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    return _THINK.sub("", text or "").strip()


def encode_prompt(tokenizer, prompt: str):
    """Tokenize text without hitting the Qwen3.5 VL processor image path.

    Unsloth patches Processor.__call__(images, text, videos). A positional
    prompt is treated as `images=` and crashes. Prefer the inner tokenizer,
    else pass `text=` explicitly.
    """
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and inner is not tokenizer:
        return inner(prompt, return_tensors="pt")
    try:
        return tokenizer(text=prompt, return_tensors="pt")
    except TypeError:
        return tokenizer(prompt, return_tensors="pt")


def _to_device(batch, device):
    if hasattr(batch, "to"):
        try:
            return batch.to(device)
        except Exception:
            pass
    out = {}
    for key, val in dict(batch).items():
        out[key] = val.to(device) if hasattr(val, "to") else val
    return out


def _decode(tokenizer, token_ids) -> str:
    inner = getattr(tokenizer, "tokenizer", None) or tokenizer
    if hasattr(inner, "decode"):
        return inner.decode(token_ids, skip_special_tokens=True)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def generate_one(model, tokenizer, scenario: dict[str, Any], max_new: int) -> str:
    prompt = prompt_for(scenario)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": scenario["instruction"]},
                ],
                **kwargs,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": scenario["instruction"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    inputs = encode_prompt(tokenizer, prompt)
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = _to_device(inputs, device)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
    }
    inner = getattr(tokenizer, "tokenizer", None) or tokenizer
    pad_id = getattr(inner, "pad_token_id", None) or getattr(inner, "eos_token_id", None)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id
    out = model.generate(**inputs, **gen_kwargs)
    seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
    input_len = inputs["input_ids"].shape[-1]
    text = _decode(tokenizer, seq[input_len:])
    text = strip_think(text.replace(IM_START, "").replace(IM_END, ""))
    return text


def run_model(
    scenarios: list[dict[str, Any]],
    model_name: str,
    adapter: str | None,
    max_seq: int,
    max_new: int,
) -> list[dict[str, Any]]:
    model, tokenizer = _load_unsloth(model_name, adapter, max_seq)
    rows = []
    for i, sc in enumerate(scenarios, 1):
        print(f"  [{i}/{len(scenarios)}] {sc['id']} {sc['topic']}", flush=True)
        text = generate_one(model, tokenizer, sc, max_new)
        rows.append(score_prediction(sc, text))
    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return rows


def print_summary(label: str, summary: dict[str, Any]) -> None:
    print(
        f"{label}: n={summary.get('n', 0)}  pass={summary.get('pass_rate', 0)}  "
        f"honorific={summary.get('honorific_pass_rate', 0)}  "
        f"s1={summary.get('s1_rate', summary.get('ai_tell_rate', 0))}  "
        f"echo={summary.get('instruction_echo_rate', 0)}  "
        f"meta={summary.get('meta_speech_rate', 0)}  "
        f"cliche={summary.get('cliche_rate', 0)}"
    )
    for track, sub in (summary.get("by_track") or {}).items():
        if not sub.get("n"):
            continue
        extra = ""
        if track == "S":
            extra = f"  hayeot={sub.get('hayeot_injection_rate', 0)}"
        else:
            extra = (
                f"  lint_fix={sub.get('lint_fix_rate', 0)}"
                f"  colloquial={sub.get('colloquial_in_record_rate', 0)}"
            )
        print(
            f"  {track}: n={sub['n']}  pass={sub.get('pass_rate', 0)}  "
            f"honorific={sub.get('honorific_pass_rate', 0)}  "
            f"s1={sub.get('s1_rate', 0)}{extra}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Natural Korean generation eval")
    p.add_argument("--scenarios", type=Path, default=EVAL_SCENARIOS)
    p.add_argument(
        "--score-only",
        action="store_true",
        help="Score gold answers in scenarios.jsonl (CPU, no model)",
    )
    p.add_argument(
        "--predictions",
        type=Path,
        help="JSONL with id + output/generation to score",
    )
    p.add_argument("--model", default=None, help="HF / Unsloth base model id")
    p.add_argument("--adapter", default=None, help="LoRA adapter directory")
    p.add_argument("--compare-base", action="store_true", help="Also run the base model")
    p.add_argument("--max-seq-length", type=int, default=1536)
    p.add_argument("--max-new-tokens", type=int, default=280)
    p.add_argument("--limit", type=int, default=0, help="Score only the first N scenarios")
    p.add_argument(
        "--track",
        choices=("S", "R"),
        default=None,
        help="Score only speech (S) or record (R) scenarios",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    scenarios = load_jsonl(args.scenarios)
    if args.track:
        scenarios = [s for s in scenarios if scenario_track(s) == args.track]
    if args.limit and args.limit > 0:
        scenarios = scenarios[: args.limit]
    report: dict[str, Any] = {"scenarios": str(args.scenarios), "n": len(scenarios)}

    if args.score_only:
        rows = []
        missing = 0
        for sc in scenarios:
            gold = sc.get("gold") or ""
            if not gold:
                missing += 1
            rows.append(score_prediction(sc, gold))
        report["gold"] = {
            "summary": summarize_run(rows),
            "rows": rows,
            "missing_gold": missing,
        }
        print_summary("gold", report["gold"]["summary"])
    elif args.predictions:
        preds = {r["id"]: r for r in load_jsonl(args.predictions)}
        rows = []
        for sc in scenarios:
            pred = preds.get(sc["id"], {})
            gen = pred.get("output") or pred.get("generation") or ""
            rows.append(score_prediction(sc, gen))
        report["predictions"] = {"summary": summarize_run(rows), "rows": rows}
        print_summary("predictions", report["predictions"]["summary"])
    elif args.model:
        if args.compare_base:
            print("base model…", flush=True)
            base_rows = run_model(
                scenarios, args.model, None, args.max_seq_length, args.max_new_tokens
            )
            report["base"] = {"summary": summarize_run(base_rows), "rows": base_rows}
            print_summary("base", report["base"]["summary"])
        print("adapter/model generate…", flush=True)
        ft_rows = run_model(
            scenarios,
            args.model,
            args.adapter,
            args.max_seq_length,
            args.max_new_tokens,
        )
        key = "adapter" if args.adapter else "model"
        report[key] = {"summary": summarize_run(ft_rows), "rows": ft_rows}
        print_summary(key, report[key]["summary"])
        if "base" in report:
            b = report["base"]["summary"].get("pass_rate", 0)
            a = report[key]["summary"].get("pass_rate", 0)
            print(f"delta pass_rate (adapter - base): {a - b:+.3f}")
            for track in ("S", "R"):
                bt = (report["base"]["summary"].get("by_track") or {}).get(track, {})
                at = (report[key]["summary"].get("by_track") or {}).get(track, {})
                if bt.get("n") or at.get("n"):
                    print(
                        f"  {track} pass {bt.get('pass_rate', 0)} → {at.get('pass_rate', 0)}"
                    )
    else:
        p.print_help()
        return 2

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
