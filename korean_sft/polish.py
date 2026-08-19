"""Polish drafts through the installed Korean skills.

Pipeline (pure over draft + metadata):
1. im-not-ai sanitize_text
2. humanize-korean deterministic AI-tell substitutions (quick-rules / playbook)
3. fluent-korean: replace em-dash, keep finite sentences
4. korean-report-style lint.fix for formal/professional only

Also exposes the real skill CLIs used by tests:
prepare_monolith_input, verify_gates, lint.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

from .generate import eul_reul, euro
from .paths import (
    POLISHED_PATH,
    RAW_PATH,
    humanize_scripts_dir,
    report_lint_path,
)

_HANGUL = re.compile(r"[가-힣]")

# Whole-sentence injected AI-tell (noun + optional josa + predicate). Drop, don't rewrite.
# Noun phrase whose last syllable is the josa (김밥은, 이번 발표는, 창가 테이블은, VPN를).
# Last-char josa is required so a hangul token cannot swallow 는/은 and miss the tell.
_NOUN_JOSA = r"(?:[가-힣A-Za-z0-9]+\s+){0,2}[가-힣A-Za-z0-9]*[은는이가을를]"
# All 5 inject kinds from generate._ai_tell_clause, plus leftover rewrites.
_STANDALONE_TELL = re.compile(
    r"(?:^|(?<=[.!?]))\s*"
    rf"{_NOUN_JOSA}\s*"
    r"(?:"
    r"논의해 볼 필요가 있다|"
    r"논의할 필요가 있다|"
    r"매우 중요하다고 할 수 있다|"
    r"중요하다|"
    r"시사하는 바가 크다|"
    r"판단되어진다|"
    r"판단된다|"
    r"강력한 경쟁력을 가지고 있다|"
    r"경쟁력이 강하다"
    r")\.?"
)

# Residue the shipped answers must not keep.
WATERMARK_RE = re.compile(r"기록번호")
STUB_JOSA_RE = re.compile(r"[은는]\.(?:\s|$)")
GLUED_TELL_RE = re.compile(
    r"(?:"
    r"논의할 필요가 있다|"
    r"논의해 볼 필요가 있다|"
    rf"{_NOUN_JOSA}\s*판단된다|"
    rf"{_NOUN_JOSA}\s*중요하다|"
    rf"{_NOUN_JOSA}\s*경쟁력이 강하다"
    r")"
)

# humanize-korean quick-rules / rewriting-playbook, removal-only.
# Longer patterns first. Applied after standalone-tell drop.
_HUMANIZE_SUBS: tuple[tuple[str, str], ...] = (
    ("바뀌는 것이라는 점에서 시사하는 바가 크다", "바뀐다"),
    ("것이라는 점에서 시사하는 바가 크다", ""),
    ("시사하는 바가 크다", ""),
    ("것이라는 점에서", ""),
    ("매우 중요하다고 할 수 있다", "중요하다"),
    ("라고 할 수 있다", "이다"),
    ("수준으로 판단되어진다", "수준이다"),
    ("판단되어진다", "이다"),
    ("되어진다", "된다"),
    ("강력한 경쟁력을 가지고 있다", "경쟁력이 강하다"),
    ("경쟁력을 가지고 있다", "경쟁력이 있다"),
    ("을 가지고 있다", "이 있다"),
    ("를 가지고 있다", "가 있다"),
    ("가지고 있다", "있다"),
    ("결론적으로, ", ""),
    ("결론적으로 ", ""),
    ("요약하면, ", ""),
    ("요약하면 ", ""),
    ("정리하자면, ", ""),
    ("정리하자면 ", ""),
    ("이를 통해 ", ""),
    ("또한, ", ""),
    ("따라서, ", ""),
    ("그러므로, ", ""),
    ("본 문서는 ", ""),
)

# fluent-korean: em-dash compresses the relation; use comma or period.
_EMDASH = re.compile(r"\s*[—–]\s*")


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # Python 3.9 dataclasses looks up cls.__module__ in sys.modules.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_SANITIZE: ModuleType | None = None
_LINT: ModuleType | None = None
_LINT_RULES: list | None = None


def sanitize_module() -> ModuleType:
    global _SANITIZE
    if _SANITIZE is None:
        _SANITIZE = _load_module(
            humanize_scripts_dir() / "sanitize_text.py", "sanitize_text"
        )
    return _SANITIZE


def lint_module() -> ModuleType:
    global _LINT
    if _LINT is None:
        _LINT = _load_module(report_lint_path(), "korean_report_lint")
    return _LINT


def _lint_rules():
    global _LINT_RULES
    if _LINT_RULES is None:
        _LINT_RULES = lint_module().load_rules()
    return _LINT_RULES


def apply_sanitize(text: str) -> str:
    san = sanitize_module()
    cleaned, _report = san.sanitize(text)
    return cleaned


def apply_humanize_rules(text: str) -> str:
    """Deterministic subset of humanize-korean quick-rules (removal-only)."""
    out = text
    out = _EMDASH.sub(". ", out)
    out = re.sub(r"\s*\(?기록번호\s*\d+\)?\.?", "", out)
    # Prefixes are glued onto the injected clause; strip them before drop.
    for src, _dst in _HUMANIZE_SUBS:
        if src.endswith(", ") or src.endswith(" "):
            out = out.replace(src, "")
    out = re.sub(r"(^|[.!?]\s*)(또한|따라서|그러므로|이를 통해|결론적으로|요약하면|정리하자면)\s+", r"\1", out)
    out = _STANDALONE_TELL.sub(" ", out)

    def _discuss(match: re.Match) -> str:
        return eul_reul(match.group(1)) + " 이야기한다"

    out = re.sub(r"([가-힣]+)[을를] 논의해 볼 필요가 있다", _discuss, out)
    out = re.sub(r"([가-힣]+)에 대해 논의해 볼 필요가 있다", _discuss, out)
    out = re.sub(
        r"([가-힣]+)에 대해 논의",
        lambda m: eul_reul(m.group(1)) + " 이야기한다",
        out,
    )
    out = out.replace("논의할 필요가 있다", "이야기한다")
    for src, dst in _HUMANIZE_SUBS:
        out = out.replace(src, dst)
    # sentence-initial leftover connectors after prefix strip
    out = re.sub(r"(^|[.!?]\s*)(또한|따라서|그러므로|이를 통해)\s+", r"\1", out)
    # leftover "Noun+josa." after a phrase delete (생활기록은.)
    out = re.sub(
        rf"(?:^|(?<=[.!?]))\s*{_NOUN_JOSA}\s*\.\s*",
        " ",
        out,
    )
    out = re.sub(r" {2,}", " ", out)
    out = re.sub(r" \.", ".", out)
    out = re.sub(r"\.\s*\.", ".", out)
    return out.strip()


def residue_problems(text: str) -> list[str]:
    """Problems a polished native answer must not keep."""
    found: list[str] = []
    if WATERMARK_RE.search(text):
        found.append("기록번호")
    if STUB_JOSA_RE.search(text):
        found.append("은./는. stub")
    glued = GLUED_TELL_RE.search(text)
    if glued:
        found.append(glued.group(0))
    return found


def apply_fluent_korean(text: str) -> str:
    """Follow fluent-korean: no em-dash; keep finite sentence endings."""
    out = _EMDASH.sub(". ", text)
    out = re.sub(r" {2,}", " ", out)
    out = re.sub(r"\.\s*\.", ".", out)
    lines = []
    for para in out.split("\n"):
        para = para.strip()
        if not para:
            lines.append("")
            continue
        if _HANGUL.search(para) and not re.search(
            r"[다요까음음임음음]$|[다요까]\.$|[다요까]!$|EOD$", para
        ):
            if para[-1] not in ".!?":
                para = para + "."
        lines.append(para)
    return "\n".join(lines).strip()


def apply_report_lint(text: str, register: str) -> str:
    """korean-report-style lint.fix — formal/professional only."""
    if register not in ("formal", "professional"):
        return text
    if register == "professional":
        text = re.sub(
            r"([가-힣]+)[을를] 다룬다",
            lambda m: euro(m.group(1)) + " 한정한다",
            text,
        )
    lint = lint_module()
    fixed, _hits = lint.fix(text, _lint_rules(), html=False)
    return fixed


_LINT_SEED = (
    "확인되었다. 정리하였다.",
    "확인됐다. 정리했다.",
    "확인되었다. 정리했다.",
    "확인됐다. 정리하였다.",
)


def polish_document(draft: str, metadata: dict[str, Any] | None = None) -> str:
    """Pure function: (draft, metadata) → native-style answer. Register preserved."""
    from .generate import to_haeyo

    meta = metadata or {}
    register = str(meta.get("register", "casual"))
    text = apply_sanitize(draft)
    text = apply_humanize_rules(text)
    text = apply_fluent_korean(text)
    text = apply_report_lint(text, register)
    for seed in _LINT_SEED:
        text = text.replace(seed, "")
    if register == "casual" and meta.get("speech_level") == "jondaet":
        text = to_haeyo(text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" \.", ".", text).strip()
    if not text:
        raise ValueError("polish produced empty text")
    return text


def genre_for(register: str, environment: str) -> str:
    if register == "professional":
        return "report"
    if register == "formal":
        return "essay"
    if environment == "online":
        return "blog"
    return "essay"


def run_prepare_monolith_input(
    text: str,
    run_dir: Path,
    genre: str = "essay",
) -> dict[str, Any]:
    """Drive the shipped im-not-ai shim (prepare_monolith_input.py)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "01_input.txt").write_text(text, encoding="utf-8")
    script = humanize_scripts_dir() / "prepare_monolith_input.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--run-dir", str(run_dir), "--genre", genre],
        check=False,
        capture_output=True,
        text=True,
    )
    metrics_path = run_dir / "00_metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(
            f"prepare_monolith_input failed rc={proc.returncode}: {proc.stderr}"
        )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def run_verify_gates(before: str, after: str, genre: str = "essay") -> dict[str, Any]:
    """Drive the shipped verify_gates.py on two texts."""
    script = humanize_scripts_dir() / "verify_gates.py"
    with tempfile.TemporaryDirectory(prefix="krs-gates-") as tmp:
        tmp_path = Path(tmp)
        before_p = tmp_path / "before.txt"
        after_p = tmp_path / "after.md"
        before_p.write_text(before, encoding="utf-8")
        after_p.write_text(after, encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--before",
                str(before_p),
                "--after",
                str(after_p),
                "--genre",
                genre,
                "--json",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    payload: dict[str, Any] = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.stdout.strip():
        try:
            payload["json"] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            pass
    return payload


def run_lint(text: str, *, fix: bool = False) -> tuple[str, list[dict[str, Any]]]:
    """Drive korean-report-style lint.py on a fixture string."""
    lint = lint_module()
    rules = _lint_rules()
    out = text
    if fix:
        out, _hits = lint.fix(text, rules, html=False)
    findings = [f.as_dict() for f in lint.lint(out, "fixture.md", rules, False)]
    return out, findings


def polish_corpus(
    src: Path = RAW_PATH,
    dst: Path = POLISHED_PATH,
    resume: bool = True,
) -> Path:
    from .generate import load_jsonl

    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = load_jsonl(src)
    existing: dict[int, dict] = {}
    if resume and dst.exists():
        for rec in load_jsonl(dst):
            existing[int(rec["id"])] = rec
    out_rows: list[dict] = []
    for rec in raw:
        doc_id = int(rec["id"])
        if doc_id in existing and existing[doc_id].get("answer"):
            out_rows.append(existing[doc_id])
            continue
        answer = polish_document(rec["body"], rec)
        polished = dict(rec)
        polished["draft"] = rec["body"]
        polished["answer"] = answer
        out_rows.append(polished)
    out_rows.sort(key=lambda r: int(r["id"]))
    with dst.open("w", encoding="utf-8") as fh:
        for rec in out_rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return dst


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Polish the raw Korean corpus")
    p.add_argument("--src", type=Path, default=RAW_PATH)
    p.add_argument("--dst", type=Path, default=POLISHED_PATH)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args(argv)
    polish_corpus(src=args.src, dst=args.dst, resume=not args.no_resume)
    print(f"polished → {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
