"""Unit tests for generation helpers (no GPU, no full corpus)."""

from __future__ import annotations

import re

from korean_sft.diversity import RELATIONS, spec_for_id
from korean_sft.generate import euro, generate_document, to_haeyo
from korean_sft.polish import polish_document, residue_problems

HANGUL = re.compile(r"[가-힣]")


def test_to_haeyo_lifts_endings_without_stacking():
    assert to_haeyo("늦었어.") == "늦었어요."
    assert "었어요요" not in to_haeyo("늦었어요.")
    assert "할게요요" not in to_haeyo("오늘 할게.")
    assert "할게요요" not in to_haeyo("오늘 할게요.")
    assert to_haeyo("천천히 해.") == "천천히 해요."
    assert "이에요" in to_haeyo("김밥이야.")


def test_euro_particles():
    assert euro("황리단길") == "황리단길로"
    assert euro("지하철") == "지하철로"
    assert euro("강남") == "강남으로"
    assert euro("학교") == "학교로"


def test_spec_covers_speech_and_generation():
    levels = set()
    gens = set()
    rels = set()
    for i in range(len(RELATIONS) * 3):
        spec = spec_for_id(i)
        levels.add(spec.speech_level)
        gens.add(spec.generation)
        rels.add(spec.relation)
        assert spec.addressee_age >= 8
        assert spec.speech_level in ("banmal", "jondaet")
    assert levels >= {"banmal", "jondaet"}
    assert gens >= {"younger_to_older", "older_to_younger", "peers"}
    assert len(rels) == len(RELATIONS)


def test_casual_jondaet_uses_haeyo_not_hapsyo():
    rec = next(
        generate_document(i)
        for i in range(48)
        if generate_document(i)["register"] == "casual"
        and generate_document(i)["speech_level"] == "jondaet"
    )
    assert "요" in rec["body"] or "세요" in rec["body"]
    assert "습니다" not in rec["body"]
    ans = polish_document(rec["body"], rec)
    assert not residue_problems(ans)


def test_casual_banmal_stays_hae():
    rec = next(
        generate_document(i)
        for i in range(48)
        if generate_document(i)["register"] == "casual"
        and generate_document(i)["speech_level"] == "banmal"
    )
    assert "습니다" not in rec["body"]
    assert "하였" not in rec["body"]
    ans = polish_document(rec["body"], rec)
    assert "습니다" not in ans
    assert not residue_problems(ans)


def test_generate_short_has_korean_and_instruction():
    rec = generate_document(3, form="short")
    assert HANGUL.search(rec["body"])
    assert rec["speech_level"] in rec["instruction"] or (
        "반말" in rec["instruction"] or "존댓말" in rec["instruction"]
    )
    assert rec["addressee"]
    assert rec["age"] != rec["addressee_age"] or rec["generation"] == "peers"
