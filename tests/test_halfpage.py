"""1-page and 5-page 1k corpora (halfpage is an alias of onepage)."""

from __future__ import annotations

import re

from korean_sft.diversity import FIVEPAGE_COUNT, ONEPAGE_COUNT
from korean_sft.generate import generate_document, load_jsonl
from korean_sft.pack import IM_END, IM_START
from korean_sft.paths import (
    FIVEPAGE_POLISHED_PATH,
    FIVEPAGE_RAW_PATH,
    FIVEPAGE_SFT_PATH,
    ONEPAGE_POLISHED_PATH,
    ONEPAGE_RAW_PATH,
    ONEPAGE_SFT_PATH,
)
from korean_sft.polish import residue_problems
from korean_sft.train import load_train_dataset

HANGUL = re.compile(r"[가-힣]")


def _check_corpus(raw_path, polished_path, sft_path, count, min_chars, min_paras):
    assert raw_path.is_file(), raw_path
    rows = load_jsonl(raw_path)
    assert len(rows) == count
    envs, regs, ages, bgs, topics = set(), set(), set(), set(), set()
    for rec in rows:
        body = rec["body"]
        assert body.strip() and HANGUL.search(body)
        paras = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        assert len(paras) >= min_paras, rec["id"]
        assert len(body) >= min_chars, (rec["id"], len(body))
        envs.add(rec["environment"])
        regs.add(rec["register"])
        ages.add(rec["age"])
        bgs.add(rec["background"])
        topics.add(rec["topic"])
    assert envs >= {"online", "offline"}
    assert regs >= {"casual", "formal", "professional"}
    assert len(ages) > 1 and len(bgs) > 1 and len(topics) > 1

    polished = load_jsonl(polished_path)
    assert len(polished) == count
    for rec in polished:
        ans = rec["answer"]
        assert ans.strip()
        assert len(ans) >= min_chars, rec["id"]
        assert not residue_problems(ans), f"id={rec['id']} {residue_problems(ans)}"
        if rec["register"] == "casual":
            assert "하였다" not in ans
        if rec.get("speech_level") == "jondaet" and rec["register"] == "casual":
            assert "요" in ans or "세요" in ans
        if rec.get("speech_level") == "banmal" and rec["register"] == "casual":
            assert "습니다" not in ans
        assert "{topic}" not in ans
        assert "확인되었다. 정리하였다." not in ans

    ds = load_train_dataset(sft_path)
    assert len(ds) == count
    by_id = {int(r["id"]): r["answer"] for r in polished}
    for i in range(min(count, 50)):
        row = ds[i]
        text = row["text"]
        assert IM_START in text and IM_END in text
        answer = by_id[int(row["id"])]
        assistant = text.split(f"{IM_START}assistant\n", 1)[1].split(IM_END, 1)[0]
        assert assistant.strip() == answer.strip()


def test_onepage_generator_length():
    rec = generate_document(7, form="onepage")
    paras = [p for p in rec["body"].split("\n\n") if p.strip()]
    assert rec["form"] == "onepage"
    assert "한 페이지" in rec["instruction"]
    assert len(paras) >= 8
    assert len(rec["body"]) >= 700


def test_fivepage_generator_has_five_sections():
    rec = generate_document(7, form="fivepage")
    assert rec["form"] == "fivepage"
    assert "다섯 페이지" in rec["instruction"]
    assert len(rec["body"]) >= 2800
    # five section titles appear
    assert rec["body"].count("\n\n") >= 15


def test_onepage_shipped_corpus():
    _check_corpus(
        ONEPAGE_RAW_PATH,
        ONEPAGE_POLISHED_PATH,
        ONEPAGE_SFT_PATH,
        ONEPAGE_COUNT,
        min_chars=650,
        min_paras=8,
    )


def test_fivepage_shipped_corpus():
    _check_corpus(
        FIVEPAGE_RAW_PATH,
        FIVEPAGE_POLISHED_PATH,
        FIVEPAGE_SFT_PATH,
        FIVEPAGE_COUNT,
        min_chars=2800,
        min_paras=18,
    )
