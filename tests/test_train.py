"""Train entry config + loader (Unsloth Qwen3.5 recipe)."""

from __future__ import annotations

from korean_sft.paths import SFT_PATH
from korean_sft.train import build_sft_config, describe_split, load_train_dataset, main as train_main


def test_sft_config_is_16bit_lora_not_qlora():
    cfg = build_sft_config(dataset_path=SFT_PATH)
    assert cfg["load_in_4bit"] is False
    assert cfg["load_in_16bit"] is True
    assert cfg["trainer"] == "SFTTrainer"
    assert cfg["sft"] is True
    assert cfg["dataset_text_field"] == "text"
    assert cfg["full_finetuning"] is False


def test_loader_sees_text_column():
    ds = load_train_dataset(SFT_PATH)
    info = describe_split(ds)
    assert info["has_text"]
    assert info["rows"] > 0
    assert info["sample_has_im_start"]
    assert info["sample_has_im_end"]


def test_train_entry_dry_run_succeeds():
    assert train_main(["--dry-run", "--dataset", str(SFT_PATH)]) == 0
