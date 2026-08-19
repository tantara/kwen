"""Unsloth Qwen3.5 SFT entry point.

Mirrors https://unsloth.ai/docs/models/qwen3.5/fine-tune :
FastLanguageModel + SFTTrainer, bf16/16-bit LoRA, not QLoRA.

`--dry-run` loads the shipped train split via `load_train_dataset` and
prints the SFT config without touching a GPU. `--try-model` attempts the
real Unsloth import/load and records failure if the stack is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT, SFT_PATH

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_OUTPUT = REPO_ROOT / "outputs_qwen35"


def load_train_dataset(path: Path | str = SFT_PATH, split: str = "train"):
    """Loader used by the train entry *and* the SFT tests."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        from datasets import load_dataset
    except ImportError:
        # stdlib fallback: list[dict] with the same `text` field
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return _ListDataset(rows)
    return load_dataset("json", data_files={"train": str(path)}, split=split)


class _ListDataset(list):
    """Minimal datasets.Dataset lookalike for environments without `datasets`."""

    @property
    def column_names(self) -> list[str]:
        return list(self[0].keys()) if self else []

    def __getitem__(self, idx):  # type: ignore[override]
        if isinstance(idx, str):
            return [row[idx] for row in self]
        return list.__getitem__(self, idx)


def build_sft_config(
    *,
    model_name: str = DEFAULT_MODEL,
    max_seq_length: int = 2048,
    max_steps: int = 100,
    output_dir: str | Path = DEFAULT_OUTPUT,
    dataset_path: str | Path = SFT_PATH,
) -> dict[str, Any]:
    """Trainer/config the entry point would pass to Unsloth + TRL."""
    return {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "load_in_4bit": False,  # Qwen3.5: QLoRA not recommended
        "load_in_16bit": True,  # bf16/16-bit LoRA
        "full_finetuning": False,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "use_gradient_checkpointing": "unsloth",
        "trainer": "SFTTrainer",
        "sft": True,
        "dataset_text_field": "text",
        "dataset_path": str(dataset_path),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 10,
        "max_steps": max_steps,
        "logging_steps": 1,
        "output_dir": str(output_dir),
        "optim": "adamw_8bit",
        "seed": 3407,
        "dataset_num_proc": 1,
    }


def describe_split(dataset) -> dict[str, Any]:
    n = len(dataset)
    cols = list(dataset.column_names) if hasattr(dataset, "column_names") else []
    sample_text = ""
    if n:
        row0 = dataset[0]
        sample_text = row0["text"] if isinstance(row0, dict) else dataset["text"][0]
    return {
        "rows": n,
        "columns": cols,
        "has_text": "text" in cols or (n > 0 and "text" in (dataset[0] or {})),
        "sample_has_im_start": "<|im_start|>" in sample_text,
        "sample_has_im_end": "<|im_end|>" in sample_text,
    }


def try_unsloth_load(cfg: dict[str, Any]) -> dict[str, Any]:
    """Attempt FastLanguageModel.from_pretrained. Returns status, never fakes success."""
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except Exception as exc:  # noqa: BLE001 — launcher must record the real error
        return {"ok": False, "stage": "import", "error": f"{type(exc).__name__}: {exc}"}
    try:
        _model, _tok = FastLanguageModel.from_pretrained(
            model_name=cfg["model_name"],
            max_seq_length=cfg["max_seq_length"],
            load_in_4bit=cfg["load_in_4bit"],
            load_in_16bit=cfg["load_in_16bit"],
            full_finetuning=cfg["full_finetuning"],
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "stage": "from_pretrained",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"ok": True, "stage": "from_pretrained"}


def run_sft(cfg: dict[str, Any], dataset) -> dict[str, Any]:
    """Full Unsloth + TRL train. Used only when --try-model and import works."""
    from unsloth import FastLanguageModel  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        load_in_16bit=cfg["load_in_16bit"],
        full_finetuning=cfg["full_finetuning"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=cfg["target_modules"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing=cfg["use_gradient_checkpointing"],
        random_state=cfg["seed"],
        max_seq_length=cfg["max_seq_length"],
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            max_seq_length=cfg["max_seq_length"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            warmup_steps=min(cfg["warmup_steps"], max(0, cfg["max_steps"] - 1)),
            max_steps=cfg["max_steps"],
            logging_steps=cfg["logging_steps"],
            output_dir=cfg["output_dir"],
            optim=cfg["optim"],
            seed=cfg["seed"],
            dataset_num_proc=cfg["dataset_num_proc"],
            dataset_text_field=cfg["dataset_text_field"],
        ),
    )
    trainer.train()
    out = str(cfg["output_dir"])
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"saved LoRA adapters to {out}")
    return {"ok": True, "trainer": "SFTTrainer", "output_dir": out}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Unsloth Qwen3.5 SFT on the local Korean set")
    p.add_argument("--dataset", type=Path, default=SFT_PATH)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load split + emit config. No model download.",
    )
    p.add_argument(
        "--try-model",
        action="store_true",
        help="Attempt Unsloth FastLanguageModel load (may fail without GPU/unsloth).",
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="Run SFTTrainer after a successful model load.",
    )
    args = p.parse_args(argv)

    cfg = build_sft_config(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        dataset_path=args.dataset,
    )
    dataset = load_train_dataset(args.dataset)
    info = describe_split(dataset)
    print("dataset:", json.dumps(info, ensure_ascii=False))
    print("config:", json.dumps(cfg, ensure_ascii=False))
    if not info["has_text"]:
        print("ERROR: train split has no `text` column", file=sys.stderr)
        return 2
    if args.dry_run and not args.try_model and not args.train:
        print("dry-run ok: SFT on `text` with 16-bit LoRA (load_in_4bit=false)")
        return 0

    if args.train:
        # Load once inside run_sft — a prior try_unsloth_load would double VRAM.
        try:
            run_sft(cfg, dataset)
        except Exception as exc:  # noqa: BLE001
            print(
                "UNSLOTH_UNAVAILABLE "
                f"stage=train error={type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 3
        print("train ok")
        return 0

    load_status = try_unsloth_load(cfg)
    print("unsloth:", json.dumps(load_status, ensure_ascii=False))
    if not load_status["ok"]:
        print(
            "UNSLOTH_UNAVAILABLE "
            f"stage={load_status['stage']} error={load_status['error']}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
