#!/usr/bin/env python3
"""Token stats for a JSONL SFT dataset using the Qwen 3.5 tokenizer.

    python3 scripts/dataset_stats.py
    python3 scripts/dataset_stats.py data/sft/train_halfpage.jsonl
    python3 scripts/dataset_stats.py data/polished/halfpage.jsonl --field answer
    python3 -m korean_sft stats --dataset data/sft/train.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from korean_sft.stats import main

if __name__ == "__main__":
    raise SystemExit(main())
