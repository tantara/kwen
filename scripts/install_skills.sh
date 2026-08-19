#!/usr/bin/env bash
# Recreate project .grok/skills/ symlinks into the vendored skill clones.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/.grok/skills"
mkdir -p "$dest/fluent-korean/output-styles"
ln -sfn ../../im-not-ai/skills/humanize-korean "$dest/humanize-korean"
ln -sfn ../../im-not-ai/skills/humanize "$dest/humanize"
ln -sfn ../../im-not-ai/skills/humanize-redo "$dest/humanize-redo"
ln -sfn ../../korean-report-skills/plugins/korean-report/skills/korean-report-style "$dest/korean-report-style"
ln -sfn ../../korean-report-skills/plugins/korean-report/skills/korean-report-doc "$dest/korean-report-doc"
ln -sfn ../../../../fluent-korean/plugins/fluent-korean/output-styles/fluent-korean.md \
    "$dest/fluent-korean/output-styles/fluent-korean.md"
ln -sfn ../../../../fluent-korean/plugins/fluent-korean/output-styles/fluent-korean-not-coding.md \
    "$dest/fluent-korean/output-styles/fluent-korean-not-coding.md"
test -f "$dest/humanize-korean/SKILL.md"
test -f "$dest/korean-report-style/SKILL.md"
test -f "$dest/fluent-korean/SKILL.md"
echo "skills installed under $dest"
