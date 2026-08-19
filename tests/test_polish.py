"""Drive shipped skill units + the project polish function."""

from __future__ import annotations

from pathlib import Path

from korean_sft.diversity import TARGET_COUNT
from korean_sft.generate import generate_document, load_jsonl
from korean_sft.paths import POLISHED_PATH, REPO_ROOT
from korean_sft.polish import (
    polish_document,
    residue_problems,
    run_lint,
    run_prepare_monolith_input,
    run_verify_gates,
)

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "ai_tell_draft.txt"


def test_prepare_monolith_input_on_fixture(tmp_path):
    draft = FIXTURE.read_text(encoding="utf-8")
    metrics = run_prepare_monolith_input(draft, tmp_path / "run", genre="essay")
    assert isinstance(metrics, dict)
    # metrics_v2 writes route_hint; older fallback may omit it but must write JSON
    assert metrics, "empty metrics JSON from prepare_monolith_input"


def test_verify_gates_on_real_texts():
    draft = FIXTURE.read_text(encoding="utf-8")
    polished = polish_document(draft, {"register": "formal"})
    result = run_verify_gates(draft, polished, genre="essay")
    assert result["returncode"] in (0, 1, 2)  # 3 = cannot judge
    assert result["returncode"] != 3


def test_lint_fix_on_formal_endings():
    text = "확인됐다. 정리했다. 변경됐으며 통합돼 있다."
    fixed, findings = run_lint(text, fix=True)
    assert "됐다" not in fixed
    assert "했다" not in fixed
    assert "되었다" in fixed or "하였다" in fixed
    assert isinstance(findings, list)


def test_polish_moves_ai_tell_fixture_in_expected_direction():
    draft = FIXTURE.read_text(encoding="utf-8")
    polished = polish_document(draft, {"register": "formal"})
    assert polished.strip()
    assert polished != draft
    # AI-tell / translationese reduced
    for bad in (
        "결론적으로",
        "시사하는 바가 크다",
        "판단되어진다",
        "매우 중요하다고 할 수 있다",
        "가지고 있다",
        "—",
    ):
        assert bad not in polished, f"still present: {bad!r}\n{polished}"
    # report lint on formal: 됐다/했다 rewritten
    assert "됐다" not in polished
    assert "했다" not in polished
    # content anchors preserved (the "발표는 매우 중요하다" inject is dropped whole)
    assert "신제품" in polished
    assert "절반" in polished
    assert not residue_problems(polished), residue_problems(polished)
    assert "것이라는 점에서" not in polished
    assert "중요하다" not in polished
    assert "경쟁력이 강하다" not in polished


def test_all_five_inject_kinds_are_dropped_not_rewritten():
    """kind = (doc_id // 3) % 5 — drop the clause, do not leave X는 중요하다."""
    # 0 discuss, 1 중요하다, 2 시사, 3 판단, 4 경쟁력
    for doc_id in (0, 5, 6, 9, 12):
        rec = generate_document(doc_id)
        ans = polish_document(rec["body"], rec)
        assert rec["body"] != ans
        problems = residue_problems(ans)
        assert not problems, f"id={doc_id} {problems}\n{ans}"
        assert "김밥은 중요하다" not in ans
        assert "연봉은 경쟁력이 강하다" not in ans


def test_all_10k_polished_answers_exist_and_nonempty():
    assert POLISHED_PATH.is_file(), f"missing {POLISHED_PATH}"
    rows = load_jsonl(POLISHED_PATH)
    assert len(rows) == TARGET_COUNT
    for rec in rows:
        ans = rec.get("answer") or ""
        assert ans.strip(), rec["id"]
        assert rec["register"] in ("casual", "formal", "professional")
        # register preserved: casual must not be upgraded to 하였
        if rec["register"] == "casual":
            assert "하였다" not in rec["answer"]
            assert "되었다" not in rec["answer"] or "됐" in rec.get("draft", "")
        if rec.get("speech_level") == "banmal" and rec["register"] == "casual":
            assert "습니다" not in rec["answer"]
        problems = residue_problems(rec["answer"])
        assert not problems, f"id={rec['id']} {problems}\n{rec['answer']}"
