"""Diversity + count checks against the *shipped* generation corpus."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from korean_sft.diversity import TARGET_COUNT
from korean_sft.generate import euro, generate_document, load_jsonl
from korean_sft.paths import RAW_PATH, SKILLS_DIR

HANGUL = re.compile(r"[가-힣]")
META_KEYS = (
    "topic",
    "environment",
    "register",
    "age",
    "background",
    "speech_level",
    "relation",
    "generation",
    "addressee",
    "addressee_age",
)


def test_euro_l_batchim_uses_ro_not_euro():
    assert euro("황리단길") == "황리단길로"
    assert euro("지하철") == "지하철로"
    assert euro("학교") == "학교로"
    assert euro("강남") == "강남으로"


def test_speech_level_follows_generation():
    seen = {generate_document(i)["speech_level"] for i in range(24)}
    assert seen >= {"banmal", "jondaet"}
    child = next(
        generate_document(i)
        for i in range(40)
        if generate_document(i)["relation"] == "child_to_parent"
        and generate_document(i)["register"] == "casual"
    )
    assert child["speech_level"] == "jondaet"
    assert "요" in child["body"] or "세요" in child["body"]
    parent = next(
        generate_document(i)
        for i in range(40)
        if generate_document(i)["relation"] == "parent_to_child"
        and generate_document(i)["register"] == "casual"
    )
    assert parent["speech_level"] == "banmal"
    assert "습니다" not in parent["body"]
    assert child["addressee_age"] > child["age"]
    assert parent["addressee_age"] < parent["age"]


def test_skills_installed():
    for name in (
        "fluent-korean",
        "korean-report-style",
        "korean-report-doc",
        "humanize-korean",
        "humanize",
        "humanize-redo",
    ):
        skill = SKILLS_DIR / name
        assert (skill / "SKILL.md").is_file(), f"missing SKILL.md for {name}"


def test_raw_corpus_exists_and_count():
    assert RAW_PATH.is_file(), f"missing shipped corpus {RAW_PATH}"
    rows = load_jsonl(RAW_PATH)
    assert len(rows) == TARGET_COUNT


def test_every_record_has_korean_body_and_metadata():
    rows = load_jsonl(RAW_PATH)
    ids = []
    for rec in rows:
        ids.append(rec["id"])
        body = rec["body"]
        assert isinstance(body, str) and body.strip(), rec["id"]
        assert HANGUL.search(body), rec["id"]
        for key in META_KEYS:
            assert key in rec and rec[key] not in (None, ""), rec
        assert rec["environment"] in ("online", "offline")
        assert rec["register"] in ("casual", "formal", "professional")
        assert rec["speech_level"] in ("banmal", "jondaet")
        assert rec["generation"] in (
            "younger_to_older",
            "older_to_younger",
            "peers",
            "service",
        )
        assert isinstance(rec["age"], int)
        assert isinstance(rec["addressee_age"], int)
    assert len(set(ids)) == TARGET_COUNT
    assert min(ids) == 0 and max(ids) == TARGET_COUNT - 1


def test_diversity_axes_have_multiple_values(tmp_path):
    rows = load_jsonl(RAW_PATH)
    hist = {key: dict(Counter(str(r[key]) for r in rows)) for key in META_KEYS}
    for key, counts in hist.items():
        assert len(counts) > 1, f"{key} is not diverse: {counts}"
    scratch = Path(
        "/var/folders/hg/3glz165524s9d4dw3_j814fc0000gn/T/"
        "grok-goal-ab0a16e177a5/implementer/diversity.json"
    )
    # Tests must not depend on scratch existing; write if parent is there.
    if scratch.parent.is_dir():
        scratch.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    assert hist["environment"].keys() >= {"online", "offline"}
    assert hist["register"].keys() >= {"casual", "formal", "professional"}
    assert hist["speech_level"].keys() >= {"banmal", "jondaet"}
    assert len(hist["relation"]) > 1
    assert len(hist["generation"]) > 1
