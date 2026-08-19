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
    assert len(rels) >= 8


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


def test_age_matches_background():
    from korean_sft.diversity import BACKGROUND_AGES, spec_for_id

    for i in range(200):
        spec = spec_for_id(i)
        lo, hi = BACKGROUND_AGES[spec.background]
        assert lo <= spec.age <= hi, (spec.background, spec.age)


def test_ritual_not_narrated_as_food():
    rec = next(
        generate_document(i)
        for i in range(160)
        if generate_document(i)["topic"] == "조문"
        and generate_document(i)["register"] == "casual"
    )
    assert "먹었어" not in rec["body"]
    assert "시켰어" not in rec["body"]


def test_to_haeyo_covers_common_banmal():
    src = "어제 갔어. 김밥 샀어. 맛있더라. 내일 할게. 그런 것 같아. 몰라."
    out = to_haeyo(src)
    for bad in ("갔어.", "샀어.", "더라.", "할게.", "같아.", "몰라."):
        assert bad not in out, (bad, out)


def test_generate_short_has_korean_and_instruction():
    rec = generate_document(3, form="short")
    assert HANGUL.search(rec["body"])
    assert rec["speech_level"] in rec["instruction"] or (
        "반말" in rec["instruction"] or "존댓말" in rec["instruction"]
    )
    assert rec["addressee"]
    assert rec["age"] != rec["addressee_age"] or rec["generation"] == "peers"
