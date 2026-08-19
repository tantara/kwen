"""Workspace paths. Skill installs live under project `.grok/skills/`."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / ".grok" / "skills"
DATA_DIR = REPO_ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "documents.jsonl"
POLISHED_PATH = DATA_DIR / "polished" / "documents.jsonl"
SFT_PATH = DATA_DIR / "sft" / "train.jsonl"
HALFPAGE_RAW_PATH = DATA_DIR / "raw" / "halfpage.jsonl"
HALFPAGE_POLISHED_PATH = DATA_DIR / "polished" / "halfpage.jsonl"
HALFPAGE_SFT_PATH = DATA_DIR / "sft" / "train_halfpage.jsonl"
ONEPAGE_RAW_PATH = DATA_DIR / "raw" / "onepage.jsonl"
ONEPAGE_POLISHED_PATH = DATA_DIR / "polished" / "onepage.jsonl"
ONEPAGE_SFT_PATH = DATA_DIR / "sft" / "train_onepage.jsonl"
FIVEPAGE_RAW_PATH = DATA_DIR / "raw" / "fivepage.jsonl"
FIVEPAGE_POLISHED_PATH = DATA_DIR / "polished" / "fivepage.jsonl"
FIVEPAGE_SFT_PATH = DATA_DIR / "sft" / "train_fivepage.jsonl"
EVAL_SCENARIOS = DATA_DIR / "eval" / "scenarios.jsonl"

FLUENT_KOREAN = SKILLS_DIR / "fluent-korean"
HUMANIZE_KOREAN = SKILLS_DIR / "humanize-korean"
HUMANIZE = SKILLS_DIR / "humanize"
HUMANIZE_REDO = SKILLS_DIR / "humanize-redo"
KOREAN_REPORT_STYLE = SKILLS_DIR / "korean-report-style"
KOREAN_REPORT_DOC = SKILLS_DIR / "korean-report-doc"

REQUIRED_SKILLS = (
    "fluent-korean",
    "korean-report-style",
    "korean-report-doc",
    "humanize-korean",
    "humanize",
    "humanize-redo",
)


def resolve_skill_root(skill_dir: Path) -> Path:
    """Walk physical parents until `.claude-plugin` (im-not-ai SKILL_ROOT)."""
    d = skill_dir.resolve()
    while d != d.parent:
        if (d / ".claude-plugin").is_dir():
            return d
        d = d.parent
    return skill_dir.resolve()


def humanize_scripts_dir() -> Path:
    return resolve_skill_root(HUMANIZE_KOREAN) / "scripts"


def report_lint_path() -> Path:
    return (KOREAN_REPORT_STYLE / "assets" / "lint.py").resolve()
