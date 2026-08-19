"""Eval metrics — CPU only, no model load."""

from korean_sft.eval import encode_prompt, strip_think
from korean_sft.eval_metrics import (
    expected_honorific,
    scenario_track,
    score_text,
    summarize_run,
)
from korean_sft.paths import REPO_ROOT


def test_ai_tell_lowers_score():
    sc = {"register": "casual", "speech_level": "jondaet", "topic": "카페"}
    bad = "결론적으로 시사하는 바가 크다. 판단되어진다."
    good = "어머니, 오늘 병원 접수하고 처방전 받아 올게요."
    assert score_text(good, sc)["naturalness"] > score_text(bad, sc)["naturalness"]
    assert score_text(bad, sc)["ai_tell"]


def test_honorific_banmal_rejects_hapsyo():
    sc = {"register": "casual", "speech_level": "banmal", "topic": "카페"}
    mix = "너는 가방 챙겨. 알겠습니다 선생님."
    ok = "야, 가방부터 챙겨. 알림장은 나중에 보자."
    assert score_text(ok, sc)["honorific_ok"]
    assert not score_text(mix, sc)["honorific_ok"]


def test_honorific_haeyo_rejects_banmal():
    sc = {"register": "casual", "speech_level": "jondaet", "topic": "병원"}
    bad = "엄마, 나 병원 다녀왔어. 처방전 받아 왔어."
    good = "엄마, 나 병원 다녀왔어요. 처방전 받아 왔어요."
    assert not score_text(bad, sc)["honorific_ok"]
    assert score_text(good, sc)["honorific_ok"]


def test_ritual_food_is_mismatch():
    sc = {"register": "casual", "speech_level": "jondaet", "topic": "조문"}
    bad = "할머니, 빈소에서 김밥 시켰어요."
    good = "할머니, 빈소에 다녀왔어요. 향 챙겼어요."
    assert score_text(bad, sc)["topic_mismatch"]
    assert not score_text(good, sc)["topic_mismatch"]


def test_expected_honorific_by_register():
    assert expected_honorific({"register": "professional", "speech_level": "jondaet"}) == "handa"
    assert expected_honorific({"register": "formal", "speech_level": "jondaet"}) == "hapsyo"
    assert expected_honorific({"register": "casual", "speech_level": "banmal"}) == "banmal"


def test_track_splits_speech_and_record():
    assert scenario_track({"register": "casual"}) == "S"
    assert scenario_track({"register": "formal"}) == "R"
    assert scenario_track({"register": "professional"}) == "R"
    assert scenario_track({"track": "S", "register": "formal"}) == "S"


def test_hayeot_injection_fails_speech_not_record():
    speech = {"register": "casual", "speech_level": "jondaet", "topic": "병원"}
    record = {"register": "professional", "speech_level": "jondaet", "topic": "코드 인수인계"}
    injected = "엄마, 병원에 다녀왔습니다. 처방전을 수령하였어요."
    report = "저장소는 백엔드이다. 배포 스크립트를 정리하였다."
    s = score_text(injected, speech)
    r = score_text(report, record)
    assert s["hayeot_injection"]
    assert not s["passed"]
    assert not r["hayeot_injection"]
    assert r["passed"]


def test_meta_and_instruction_echo_fail():
    sc = {
        "register": "casual",
        "speech_level": "banmal",
        "topic": "카페",
        "instruction": "38세 부모가 10세 아이에게 반말로 말한다. 존댓말 쓰지 마라.",
    }
    echo = "38세 부모가 10세 아이에게 반말로 말한다. 가방 챙겨."
    meta = "당신은 원어민 한국어 화자처럼 글을 쓴다. 가방 챙겨."
    ok = "가방이랑 알림장 챙겨. 하원할 때 놓고 나오면 안 돼."
    assert score_text(echo, sc)["instruction_echo"]
    assert score_text(meta, sc)["meta_speech"]
    assert not score_text(echo, sc)["passed"]
    assert score_text(ok, sc)["passed"]


def test_record_rejects_haeyo_and_lint_haetda():
    sc = {"register": "professional", "speech_level": "jondaet", "topic": "성과 평가"}
    mixed = "목표는 분기 매출 성장이에요. 실적은 낮았어요."
    haetda = "목표를 정리했다. 실적을 기재했다."
    good = "목표는 분기 매출 12% 성장이다. 실적은 9%에 그쳤다. 다음 분기 채널 믹스를 조정한다."
    assert not score_text(mixed, sc)["honorific_ok"]
    assert score_text(haetda, sc)["lint_fix"]
    assert not score_text(haetda, sc)["passed"]
    assert score_text(good, sc)["passed"]


def test_scenarios_file_exists_and_covers_axes():
    path = REPO_ROOT / "data" / "eval" / "scenarios.jsonl"
    assert path.is_file()
    import json

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) >= 20
    levels = {r["speech_level"] for r in rows}
    regs = {r["register"] for r in rows}
    gens = {r["generation"] for r in rows}
    assert levels >= {"banmal", "jondaet"}
    assert regs >= {"casual", "formal", "professional"}
    assert "younger_to_older" in gens and "older_to_younger" in gens
    assert all(r.get("gold") for r in rows)
    tracks = {r["track"] for r in rows}
    assert tracks == {"S", "R"}
    assert sum(1 for r in rows if r["track"] == "R") >= 4


def test_gold_ceiling_is_high():
    import json

    path = REPO_ROOT / "data" / "eval" / "scenarios.jsonl"
    scenarios = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [score_text(sc["gold"], sc) for sc in scenarios]
    s = summarize_run(rows)
    assert s["n"] == len(scenarios)
    assert s["pass_rate"] == 1.0
    assert s["by_track"]["S"]["pass_rate"] == 1.0
    assert s["by_track"]["R"]["pass_rate"] == 1.0
    assert s["honorific_pass_rate"] == 1.0
    assert s["s1_rate"] == 0
    assert s["topic_mismatch_rate"] == 0
    assert s["instruction_echo_rate"] == 0


def test_strip_think_block():
    raw = "<think>\nplan\n</think>\n엄마, 다녀왔어요."
    assert strip_think(raw) == "엄마, 다녀왔어요."


def test_encode_prompt_uses_text_kwarg_not_images():
    class FakeProcessor:
        def __init__(self):
            self.calls = []

        def __call__(self, images=None, text=None, videos=None, **kwargs):
            self.calls.append({"images": images, "text": text, "kwargs": kwargs})
            if images is not None and text is None:
                raise AssertionError("positional text must not be treated as images")
            return {"input_ids": [[1, 2, 3]]}

    proc = FakeProcessor()
    out = encode_prompt(proc, "안녕하세요")
    assert out["input_ids"] == [[1, 2, 3]]
    assert proc.calls and proc.calls[0]["text"] == "안녕하세요"
    assert proc.calls[0]["images"] is None


def test_summarize_run():
    sc = {"register": "casual", "speech_level": "jondaet", "topic": "병원"}
    rows = [
        score_text("어머니, 다녀올게요.", sc),
        score_text("결론적으로 시사하는 바가 크다.", sc),
    ]
    s = summarize_run(rows)
    assert s["n"] == 2
    assert 0 <= s["mean_naturalness"] <= 100
    assert 0 <= s["pass_rate"] <= 1
