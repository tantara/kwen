# Native Korean SFT for Qwen3.5 (Unsloth)

Fine-tune [Qwen3.5](https://unsloth.ai/docs/models/qwen3.5/fine-tune) so it
writes **native Korean speaker** text. Drafts are generated across topics,
online/offline settings, casual/formal/professional registers, and many ages
and backgrounds, then polished with the installed skills.

## Installed skills

Project `.grok/skills/` (symlinks into the clones):

| Name | Source |
|---|---|
| `fluent-korean` | wrapped output-style from `fluent-korean/` |
| `korean-report-style`, `korean-report-doc` | `korean-report-skills/` |
| `humanize-korean`, `humanize`, `humanize-redo` | `im-not-ai/` |

## Data layout

| Path | What |
|---|---|
| `data/raw/documents.jsonl` | 10k short drafts + metadata |
| `data/polished/documents.jsonl` | same rows with `draft` + polished `answer` |
| `data/sft/train.jsonl` | Unsloth SFT split: `text` column is Qwen chat-templated |
| `data/raw/onepage.jsonl` | 1k one-page drafts |
| `data/polished/onepage.jsonl` | polished one-page answers |
| `data/sft/train_onepage.jsonl` | Qwen-templated SFT split, 1 page × 1k |
| `data/raw/fivepage.jsonl` | 1k five-page drafts (5 sections) |
| `data/polished/fivepage.jsonl` | polished five-page answers |
| `data/sft/train_fivepage.jsonl` | Qwen-templated SFT split, 5 pages × 1k |

Each SFT row is `system` + instruction → **polished** answer, formatted with
`<|im_start|>` / `<|im_end|>` turn markers.

Speech level (`speech_level`: `banmal` | `jondaet`) follows generation
(child→parent 존댓말, parent→child 반말, close peers 반말, first-meet peers 존댓말, etc.).

## Setup (uv, no GPU required)

```bash
uv sync                  # pytest + tokenizers (CPU)
uv run pytest
uv run python -m korean_sft train --dry-run
```

GPU training later:

```bash
uv sync --extra train
uv run python -m korean_sft train --try-model --train --max-steps 100
```

## Pipeline

```bash
python3 -m korean_sft generate          # resume-safe 10k raw docs
python3 -m korean_sft polish            # skill-routed polish
python3 -m korean_sft pack              # Qwen chat template → text
python3 -m korean_sft train --dry-run   # load split + print 16-bit LoRA config

python3 -m korean_sft pipeline --length onepage    # 1k × 1 page
python3 -m korean_sft pipeline --length fivepage   # 1k × 5 pages
python3 -m korean_sft train --dry-run --dataset data/sft/train_onepage.jsonl
```

Or `python3 -m korean_sft pipeline`.

## Dataset token stats (Qwen 3.5 tokenizer)

```bash
python3 scripts/dataset_stats.py data/sft/train_halfpage.jsonl
python3 -m korean_sft stats data/sft/train.jsonl
python3 scripts/dataset_stats.py data/polished/halfpage.jsonl --field answer
```

## Train (Unsloth Qwen3.5)

Recipe matches the official guide: `FastLanguageModel` + `SFTTrainer`,
**16-bit LoRA**, `load_in_4bit=False` (QLoRA is not recommended for Qwen3.5).
Needs `transformers` v5. This machine has no GPU; `--dry-run` is the
supported path until you `uv sync --extra train` on a GPU host.

```bash
uv run python -m korean_sft train --dry-run --dataset data/sft/train.jsonl
uv run python -m korean_sft train --dry-run --dataset data/sft/train_onepage.jsonl
uv run python -m korean_sft train --dry-run --dataset data/sft/train_fivepage.jsonl
```

## Eval (natural Korean)

24 held-out scenarios in two tracks, scored with the cloned-repo gates (not one blended naturalness number):

| Track | Scenarios | Target | Gates |
|---|---|---|---|
| **S** speech | 반말 / 해요체 / 창구 | fluent-korean + im-not-ai | honorific mix, `하였` injection, S1 taxonomy, C-11 연결어미 쉼표, instruction echo |
| **R** record | 한다체 / 합니다체 보고 | korean-report-style + im-not-ai D/A | 해요/반말 in a record, lint `했다`→`하였다`, S1, echo |

Gold replies are a CPU ceiling (`pass_rate` should be 1.0). GPU generates with the base model and LoRA adapters, then scores the same axes per track.

```bash
uv run python -m korean_sft eval --score-only
uv run python -m korean_sft eval --score-only --track S
uv run python -m korean_sft eval --score-only --track R
# on GPU (base vs LoRA):
python -m korean_sft eval --model Qwen/Qwen3.5-4B \
  --adapter outputs/qwen35-4b-onepage --compare-base \
  --out reports/eval-4b.json
bash scripts/remote_eval.sh
```

## Tests

```bash
uv run pytest
```
