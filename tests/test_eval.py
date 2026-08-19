"""Eval metrics — CPU only, no model load."""

from korean_sft.eval import encode_prompt, strip_think
from korean_sft.eval_metrics import expected_honorific, score_text, summarize_run
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


def test_gold_ceiling_is_high():
    import json

    path = REPO_ROOT / "data" / "eval" / "scenarios.jsonl"
    scenarios = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [score_text(sc["gold"], sc) for sc in scenarios]
    s = summarize_run(rows)
    assert s["n"] == len(scenarios)
    assert s["mean_naturalness"] >= 90
    assert s["honorific_pass_rate"] >= 0.9
    assert s["ai_tell_rate"] == 0
    assert s["topic_mismatch_rate"] == 0


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
