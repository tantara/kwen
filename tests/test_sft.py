"""SFT JSONL loaded by the same function the train entry uses."""

from __future__ import annotations

from korean_sft.diversity import TARGET_COUNT
from korean_sft.generate import load_jsonl
from korean_sft.pack import IM_END, IM_START
from korean_sft.paths import POLISHED_PATH, SFT_PATH
from korean_sft.polish import residue_problems
from korean_sft.train import load_train_dataset


def test_sft_file_has_10000_templated_rows():
    ds = load_train_dataset(SFT_PATH)
    assert len(ds) == TARGET_COUNT
    cols = list(ds.column_names) if hasattr(ds, "column_names") else list(ds[0].keys())
    assert "text" in cols
    polished = {int(r["id"]): r["answer"] for r in load_jsonl(POLISHED_PATH)}
    drafts = {int(r["id"]): r["draft"] for r in load_jsonl(POLISHED_PATH)}
    for i in range(TARGET_COUNT):
        row = ds[i]
        text = row["text"]
        assert text and text.strip()
        assert IM_START in text and IM_END in text
        assert f"{IM_START}user" in text
        assert f"{IM_START}assistant" in text
        answer = polished[int(row["id"])] if "id" in row else polished[i]
        assert answer in text
        # assistant span is the polished document, not the raw draft
        draft = drafts[int(row["id"])] if "id" in row else drafts[i]
        if draft != answer:
            # the raw AI-tell draft must not be the assistant body
            assistant = text.split(f"{IM_START}assistant\n", 1)[1]
            assistant = assistant.split(IM_END, 1)[0]
            assert assistant.strip() == answer.strip()
            assert assistant.strip() != draft.strip()
        problems = residue_problems(answer)
        assert not problems, f"id={row.get('id', i)} {problems}"
        assert "기록번호" not in text
        if i >= 20 and i % 487 != 0:
            # full 10k equality already covered by count; sample rest cheaply
            continue
