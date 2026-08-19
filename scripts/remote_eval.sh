#!/usr/bin/env bash
# GPU eval: base vs LoRA on realistic Korean scenarios.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/hf}"
export TOKENIZERS_PARALLELISM=false
export UNSLOTH_SKIP_TORCHVISION_CHECK=1
cd /home/jedi/kwen
source .venv/bin/activate
mkdir -p reports

echo "[$(date -Is)] eval 0.8B base vs adapter"
.venv/bin/python -m korean_sft eval \
  --model Qwen/Qwen3.5-0.8B \
  --adapter outputs/qwen35-0.8b-10k \
  --compare-base \
  --max-new-tokens 220 \
  --out reports/eval-0.8b.json

echo "[$(date -Is)] eval 4B base vs adapter"
.venv/bin/python -m korean_sft eval \
  --model Qwen/Qwen3.5-4B \
  --adapter outputs/qwen35-4b-onepage \
  --compare-base \
  --max-new-tokens 280 \
  --out reports/eval-4b.json

echo "[$(date -Is)] done"
ls -la reports
.venv/bin/python - << "PY"
import json
from pathlib import Path
for p in Path("reports").glob("eval-*.json"):
    r = json.loads(p.read_text())
    print("====", p.name)
    for k in ("base", "adapter", "model"):
        if k in r and "summary" in r[k]:
            s = r[k]["summary"]
            print(f"  {k}: naturalness={s['mean_naturalness']} honorific={s['honorific_pass_rate']} ai_tell={s['ai_tell_rate']} mismatch={s['topic_mismatch_rate']}")
PY
