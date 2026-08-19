"""Token stats for a JSONL SFT dataset using the Qwen 3.5 tokenizer.

Reports number of samples, mean tokens per sample, and total tokens.
Default field is `text` (the Unsloth train column).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from .paths import HALFPAGE_SFT_PATH, SFT_PATH

DEFAULT_TOKENIZER = "Qwen/Qwen3.5-0.8B"


def load_texts(path: Path, field: str) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if field not in rec or rec[field] is None:
                raise KeyError(f"{path}:{i} missing field {field!r}")
            texts.append(str(rec[field]))
    if not texts:
        raise ValueError(f"no rows in {path}")
    return texts


class _RawTokenizer:
    """tokenizers.Tokenizer wrapper when transformers is too old for Qwen3.5."""

    def __init__(self, inner, name: str):
        self._inner = inner
        self.name_or_path = name
        self.vocab_size = inner.get_vocab_size()

    def encode_batch_ids(self, texts: list[str]) -> list[list[int]]:
        return [enc.ids for enc in self._inner.encode_batch(texts)]


def load_qwen35_tokenizer(name: str):
    """Load the official Qwen 3.5 tokenizer (shared across the 3.5 family)."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        return tok
    except Exception:
        pass
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit(
            "need transformers (v5) or huggingface_hub+tokenizers to load "
            f"the Qwen 3.5 tokenizer ({name})"
        ) from exc
    try:
        path = hf_hub_download(repo_id=name, filename="tokenizer.json")
    except Exception as exc:
        raise SystemExit(
            f"failed to download {name} tokenizer.json: {type(exc).__name__}: {exc}"
        ) from exc
    return _RawTokenizer(Tokenizer.from_file(path), name)


def count_tokens(tokenizer, texts: list[str], batch_size: int = 64) -> list[int]:
    lengths: list[int] = []
    raw = isinstance(tokenizer, _RawTokenizer)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        if raw:
            lengths.extend(len(ids) for ids in tokenizer.encode_batch_ids(batch))
            continue
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def summarize(lengths: list[int]) -> dict:
    n = len(lengths)
    total = int(sum(lengths))
    return {
        "num_samples": n,
        "total_tokens": total,
        "avg_tokens_per_sample": (total / n) if n else 0.0,
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
        "median_tokens": float(statistics.median(lengths)) if lengths else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Qwen 3.5 token stats for a JSONL dataset")
    p.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=None,
        help="JSONL path (default: half-page SFT if present, else 10k SFT)",
    )
    p.add_argument("--field", default="text", help="JSON field to tokenize (default: text)")
    p.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"HF tokenizer id (default: {DEFAULT_TOKENIZER})",
    )
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)

    path = args.dataset
    if path is None:
        path = HALFPAGE_SFT_PATH if HALFPAGE_SFT_PATH.is_file() else SFT_PATH
    path = path.resolve()
    if not path.is_file():
        print(f"dataset not found: {path}", file=sys.stderr)
        return 2

    texts = load_texts(path, args.field)
    tokenizer = load_qwen35_tokenizer(args.tokenizer)
    lengths = count_tokens(tokenizer, texts, batch_size=args.batch_size)
    stats = summarize(lengths)
    tok_name = getattr(tokenizer, "name_or_path", args.tokenizer)
    vocab = getattr(tokenizer, "vocab_size", None)
    print(f"dataset:   {path}")
    print(f"field:     {args.field}")
    print(f"tokenizer: {tok_name}  vocab_size={vocab}")
    print(f"samples:   {stats['num_samples']}")
    print(f"avg tokens / sample: {stats['avg_tokens_per_sample']:.2f}")
    print(f"total tokens:        {stats['total_tokens']}")
    print(
        f"min / median / max:  {stats['min_tokens']} / "
        f"{stats['median_tokens']:.1f} / {stats['max_tokens']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
