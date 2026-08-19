"""Track-S / track-R scores for Korean generation (CPU, no GPU).

Track S is speech (반말 / 해요체 / 창구 합쇼). Track R is record
(한다체 / 합니다체 보고). Axes come from fluent-korean, im-not-ai, and
korean-report-style. There is no single naturalness number that means
the same thing on both tracks; `passed` and `by_track` are the headline.
"""

from __future__ import annotations

import re
import sys
from typing import Any

from .paths import HUMANIZE_KOREAN
from .polish import residue_problems

HANGUL = re.compile(r"[가-힣]")
BANMAL_END = re.compile(
    r"(었어|았어|였어|졌어|갔어|왔어|봤어|샀어|줬어|했어|됐어|더라|할게|같아|아니야|몰라)(?:[\s.!?…]|$)"
)
HAEYO_END = re.compile(r"(어요|아요|예요|이에요|세요|죠)(?:[\s.!?…]|$)")
HAPSYO_END = re.compile(r"(습니다|ㅂ니다|입니다)(?:[\s.!?…]|$)")
HANDA_END = re.compile(r"(한다|이다|하였다|되었다)(?:[\s.!?…]|$)")
HAYEOT = re.compile(r"하였")
# C-11: 연결어미 + comma. Exclude 이며, which is a copula, not 며.
C11_COMMA = re.compile(r"(?:고|지만|면서|아서|어서)\s*,|(?<!이)며\s*,")
THIRD_PERSON_PROMPT = re.compile(
    r"\d+세.{0,20}(가|이).{0,24}(말한다|전한다|보고한다|여쭤|안내한다|적는다)"
)

# im-not-ai D-1 / A-7 / A-8 S1 surface forms (taxonomy + checks.py).
S1_LEXICON: tuple[tuple[str, str], ...] = (
    ("D-1", "결론적으로"),
    ("D-1", "시사하는 바가"),
    ("D-1", "요약하면"),
    ("D-1", "이를 통해"),
    ("D-1", "그러므로"),
    ("D-2", "매우 중요하다고 할 수 있다"),
    ("A-7", "가지고 있다"),
    ("A-8", "되어진다"),
    ("A-8", "판단되어"),
)
# Dataset-template clichés (kwen SFT leakage) + im-not-ai D injection list.
CLICHE = (
    "옆 테이블에서",
    "마음이 안 놓",
    "번이나 다시 확인",
    "환승이 한 번 더",
    "기록적인 성과",
    "괄목할 만한",
    "시사하는 바가 크다",
    "의미가 크다",
)
META_LEXICON = (
    "당신은 원어민",
    "반말 쓰지 마라",
    "존댓말 쓰지 마라",
    "반말 금지",
    "존댓말 금지",
    "해요체로",
    "합니다체로",
    "한다체로",
    "반말로 말한다",
    "존댓말로 말한다",
    "구어체·반말 금지",
    "구어 금지",
    "as an AI",
    "The user",
    "I should",
)
FOOD_VERB = re.compile(r"(먹었어|시켰어|김밥|떡볶이|치킨)")
RITUAL_TOPICS = {"조문", "명절 차례", "결혼식 축사", "병문안"}

_V1 = None
_V2 = None


def _humanize_metrics():
    """Load im-not-ai metrics.py / metrics_v2.py from the cloned skill."""
    global _V1, _V2
    if _V1 is not None:
        return _V1, _V2
    ref = (HUMANIZE_KOREAN / "references").resolve()
    if str(ref) not in sys.path:
        sys.path.insert(0, str(ref))
    import metrics as v1  # type: ignore
    import metrics_v2 as v2  # type: ignore

    _V1, _V2 = v1, v2
    return v1, v2


def scenario_track(scenario: dict[str, Any] | None) -> str:
    """S = speech, R = record. Explicit `track` wins."""
    scenario = scenario or {}
    listed = scenario.get("track")
    if listed in ("S", "R"):
        return listed
    if scenario.get("register") in ("formal", "professional"):
        return "R"
    return "S"


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


def honorific_ok(text: str, expected: str, track: str = "S") -> bool:
    """expected: banmal | haeyo | hapsyo | handa."""
    p = honorific_profile(text)
    if expected == "banmal":
        return p["hapsyo"] == 0 and p["haeyo"] <= 1
    if expected == "haeyo":
        # Allow one 합니다체 인사말 (뵙겠습니다, 감사합니다) mixed into 해요체.
        return p["banmal"] == 0 and p["hapsyo"] <= 1
    if expected == "hapsyo":
        if track == "R":
            return p["banmal"] == 0 and p["haeyo"] == 0
        return p["banmal"] == 0
    if expected == "handa":
        return p["banmal"] == 0 and p["haeyo"] == 0
    return True


def hayeot_injection(text: str, expected: str, track: str) -> bool:
    """im-not-ai `hayeot_injection`: 하였 must not appear in 해요/반말 speech."""
    if track != "S":
        return False
    if expected in ("hapsyo", "handa"):
        return False
    return bool(HAYEOT.search(text or ""))


def colloquial_in_record(text: str, track: str) -> bool:
    """해요/반말 endings inside a 한다/합니다 record."""
    if track != "R":
        return False
    p = honorific_profile(text)
    return p["banmal"] > 0 or p["haeyo"] > 0


def topic_mismatch(text: str, topic: str) -> bool:
    if topic in RITUAL_TOPICS and FOOD_VERB.search(text):
        return True
    return False


def ending_comma_hits(text: str) -> int:
    """im-not-ai C-11 (연결어미 뒤 쉼표), excluding copula 이며,."""
    return len(C11_COMMA.findall(text or ""))


def s1_hits(text: str) -> list[str]:
    """im-not-ai S1 / strong S2 surface hits. Ids are taxonomy codes."""
    text = text or ""
    found: list[str] = []
    seen: set[str] = set()

    def add(code: str) -> None:
        if code not in seen:
            seen.add(code)
            found.append(code)

    for code, span in S1_LEXICON:
        if span in text:
            add(code)
    if ending_comma_hits(text):
        add("C-11")
    try:
        _v1, v2 = _humanize_metrics()
        if v2.double_passive_count(text) > 0:
            add("A-8")
        if v2.have_make_literal_count(text) > 0:
            add("A-7")
        if v2.by_passive_count(text) > 0:
            add("A-9")
        if v2.double_particle_count(text) > 0:
            add("A-19")
        if _v1.conclusion_pivot_count(text) > 0:
            add("D-1")
    except Exception:
        pass
    return found


def cliche_hits(text: str) -> list[str]:
    return [p for p in CLICHE if p in (text or "")]


def meta_speech_hits(text: str) -> list[str]:
    return [p for p in META_LEXICON if p in (text or "")]


def instruction_echo_hits(text: str, instruction: str) -> list[str]:
    """Fail if the model restates the prompt instead of speaking in-character."""
    text = text or ""
    instruction = instruction or ""
    hits: list[str] = []
    if THIRD_PERSON_PROMPT.search(text):
        hits.append("third_person_prompt")
    prefix = instruction.strip()
    if len(prefix) >= 18 and prefix[:18] in text:
        hits.append("instruction_prefix")
    for sent in re.split(r"[.!?]", instruction):
        sent = sent.strip()
        if len(sent) >= 14 and sent in text:
            hits.append(sent[:32])
    return hits


def report_lint_fixes(text: str) -> list[str]:
    """korean-report-style 고침-tier hits (했다→하였다). Track R only."""
    from .polish import _lint_rules, lint_module

    lint = lint_module()
    findings = lint.lint(text or "", "eval.md", _lint_rules(), False)
    return [f.found for f in findings if getattr(f, "tier", "") == "고침"]


def score_text(text: str, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per-text axes. `passed` is the gate; `naturalness` is a compatibility score."""
    scenario = scenario or {}
    text = text or ""
    track = scenario_track(scenario)
    residue = residue_problems(text)
    s1 = s1_hits(text)
    cliches = cliche_hits(text)
    meta = meta_speech_hits(text)
    echo = instruction_echo_hits(text, str(scenario.get("instruction") or ""))
    expected = expected_honorific(scenario) if scenario else ""
    hon_ok = honorific_ok(text, expected, track) if expected else True
    mismatch = topic_mismatch(text, str(scenario.get("topic", "")))
    ratio = hangul_ratio(text)
    hayeot = hayeot_injection(text, expected, track)
    colloquial = colloquial_in_record(text, track)
    lint_fixes = report_lint_fixes(text) if track == "R" else []
    placeholder = "{topic}" in text
    short = len(text.strip()) < 18

    fails: list[str] = []
    if not hon_ok:
        fails.append("honorific")
    if s1:
        fails.append("s1")
    if residue:
        fails.append("residue")
    if mismatch:
        fails.append("topic_mismatch")
    if meta:
        fails.append("meta_speech")
    if echo:
        fails.append("instruction_echo")
    if placeholder:
        fails.append("unfilled_slot")
    if hayeot:
        fails.append("hayeot_injection")
    if colloquial:
        fails.append("colloquial_in_record")
    if lint_fixes:
        fails.append("lint_fix")
    if cliches:
        fails.append("cliche")
    if short:
        fails.append("short")
    if ratio < 0.28:
        fails.append("low_hangul")

    score = 100.0
    score -= 20.0 * sum(
        1
        for k in (
            "honorific",
            "s1",
            "meta_speech",
            "instruction_echo",
            "hayeot_injection",
            "colloquial_in_record",
        )
        if k in fails
    )
    score -= min(20.0, 8.0 * len(cliches))
    score -= 15.0 if mismatch else 0.0
    score -= 15.0 if short else 0.0
    score -= 15.0 if ratio < 0.28 else 0.0
    score -= 25.0 if placeholder else 0.0
    score -= min(20.0, 8.0 * len(residue))
    score -= 10.0 if lint_fixes else 0.0
    score = max(0.0, min(100.0, score))

    return {
        "track": track,
        "chars": len(text),
        "hangul_ratio": round(ratio, 3),
        "residue": residue,
        "ai_tell": s1,
        "s1": s1,
        "cliche": cliches,
        "meta_speech": meta,
        "instruction_echo": echo,
        "ending_comma": ending_comma_hits(text),
        "hayeot_injection": hayeot,
        "colloquial_in_record": colloquial,
        "lint_fix": lint_fixes,
        "honorific": honorific_profile(text),
        "honorific_expected": expected,
        "honorific_ok": hon_ok,
        "topic_mismatch": mismatch,
        "fails": fails,
        "passed": not fails,
        "naturalness": round(score, 1),
    }


def _rate(rows: list[dict[str, Any]], pred) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for r in rows if pred(r)) / len(rows), 3)


def _track_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0, "pass_rate": 0.0, "mean_naturalness": 0.0}
    return {
        "n": n,
        "pass_rate": _rate(rows, lambda r: r.get("passed")),
        "mean_naturalness": round(sum(r["naturalness"] for r in rows) / n, 2),
        "honorific_pass_rate": _rate(rows, lambda r: r.get("honorific_ok")),
        "s1_rate": _rate(rows, lambda r: r.get("s1") or r.get("ai_tell")),
        "ai_tell_rate": _rate(rows, lambda r: r.get("s1") or r.get("ai_tell")),
        "cliche_rate": _rate(rows, lambda r: r.get("cliche")),
        "meta_speech_rate": _rate(rows, lambda r: r.get("meta_speech")),
        "instruction_echo_rate": _rate(rows, lambda r: r.get("instruction_echo")),
        "hayeot_injection_rate": _rate(rows, lambda r: r.get("hayeot_injection")),
        "colloquial_in_record_rate": _rate(
            rows, lambda r: r.get("colloquial_in_record")
        ),
        "lint_fix_rate": _rate(rows, lambda r: r.get("lint_fix")),
        "topic_mismatch_rate": _rate(rows, lambda r: r.get("topic_mismatch")),
        "ending_comma_rate": _rate(rows, lambda r: r.get("ending_comma")),
        "mean_hangul_ratio": round(sum(r.get("hangul_ratio", 0) for r in rows) / n, 3),
    }


def summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "pass_rate": 0.0, "mean_naturalness": 0.0, "by_track": {}}
    summary = _track_summary(rows)
    summary["by_track"] = {
        t: _track_summary([r for r in rows if r.get("track") == t]) for t in ("S", "R")
    }
    return summary
