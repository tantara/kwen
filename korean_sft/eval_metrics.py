"""Automatic scores for natural Korean generation (no GPU)."""

from __future__ import annotations

import re
from typing import Any

from .polish import residue_problems

HANGUL = re.compile(r"[가-힣]")
BANMAL_END = re.compile(
    r"(었어|았어|였어|졌어|갔어|왔어|봤어|샀어|줬어|했어|됐어|더라|할게|같아|아니야|몰라)(?:[\s.!?…]|$)"
)
HAEYO_END = re.compile(r"(어요|아요|예요|이에요|세요|죠)(?:[\s.!?…]|$)")
HAPSYO_END = re.compile(r"(습니다|ㅂ니다|입니다)(?:[\s.!?…]|$)")
HANDA_END = re.compile(r"(한다|이다|하였다|되었다)(?:[\s.!?…]|$)")
AI_TELL = (
    "결론적으로",
    "시사하는 바가",
    "판단되어",
    "논의해 볼 필요",
    "매우 중요하다고 할 수 있다",
    "가지고 있다",
    "이를 통해",
    "요약하면",
)
CLICHE = (
    "옆 테이블에서",
    "마음이 안 놓",
    "번이나 다시 확인",
    "환승이 한 번 더",
)
FOOD_VERB = re.compile(r"(먹었어|시켰어|김밥|떡볶이|치킨)")
RITUAL_TOPICS = {"조문", "명절 차례", "결혼식 축사", "병문안"}


def hangul_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if "가" <= ch <= "힣") / len(text)


def honorific_profile(text: str) -> dict[str, int]:
    return {
        "banmal": len(BANMAL_END.findall(text)),
        "haeyo": len(HAEYO_END.findall(text)),
        "hapsyo": len(HAPSYO_END.findall(text)),
        "handa": len(HANDA_END.findall(text)),
    }


def honorific_ok(text: str, expected: str) -> bool:
    """expected: banmal | haeyo | hapsyo | handa."""
    p = honorific_profile(text)
    if expected == "banmal":
        return p["hapsyo"] == 0 and p["haeyo"] <= 1
    if expected == "haeyo":
        # Allow one 합니다체 인사말 (뵙겠습니다, 감사합니다) mixed into 해요체.
        return p["banmal"] == 0 and p["hapsyo"] <= 1
    if expected == "hapsyo":
        return p["banmal"] == 0
    if expected == "handa":
        return p["banmal"] == 0
    return True


def expected_honorific(scenario: dict[str, Any]) -> str:
    register = scenario.get("register", "casual")
    level = scenario.get("speech_level", "jondaet")
    if scenario.get("generation") == "service":
        return "hapsyo"
    if register == "professional":
        return "handa"
    if register == "formal":
        return "hapsyo"
    if level == "banmal":
        return "banmal"
    return "haeyo"


def topic_mismatch(text: str, topic: str) -> bool:
    if topic in RITUAL_TOPICS and FOOD_VERB.search(text):
        return True
    return False


def score_text(text: str, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return per-text metrics. `naturalness` is 0–100, higher is better."""
    scenario = scenario or {}
    text = text or ""
    residue = residue_problems(text)
    tells = [p for p in AI_TELL if p in text]
    cliches = [p for p in CLICHE if p in text]
    expected = expected_honorific(scenario) if scenario else ""
    hon_ok = honorific_ok(text, expected) if expected else True
    mismatch = topic_mismatch(text, str(scenario.get("topic", "")))
    ratio = hangul_ratio(text)
    short = len(text.strip()) < 18

    score = 100.0
    score -= min(45.0, 12.0 * (len(residue) + len(tells)))
    if not hon_ok:
        score -= 20.0
    if ratio < 0.28:
        score -= 15.0
    if cliches:
        score -= min(15.0, 5.0 * len(cliches))
    if mismatch:
        score -= 15.0
    if short:
        score -= 15.0
    if "{topic}" in text:
        score -= 25.0
    score = max(0.0, min(100.0, score))

    return {
        "chars": len(text),
        "hangul_ratio": round(ratio, 3),
        "residue": residue,
        "ai_tell": tells,
        "cliche": cliches,
        "honorific": honorific_profile(text),
        "honorific_expected": expected,
        "honorific_ok": hon_ok,
        "topic_mismatch": mismatch,
        "naturalness": round(score, 1),
    }


def summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mean_naturalness": 0.0}
    n = len(rows)
    mean_nat = sum(r["naturalness"] for r in rows) / n
    return {
        "n": n,
        "mean_naturalness": round(mean_nat, 2),
        "honorific_pass_rate": round(
            sum(1 for r in rows if r["honorific_ok"]) / n, 3
        ),
        "ai_tell_rate": round(sum(1 for r in rows if r["ai_tell"]) / n, 3),
        "cliche_rate": round(sum(1 for r in rows if r["cliche"]) / n, 3),
        "topic_mismatch_rate": round(
            sum(1 for r in rows if r["topic_mismatch"]) / n, 3
        ),
        "mean_hangul_ratio": round(
            sum(r["hangul_ratio"] for r in rows) / n, 3
        ),
    }
