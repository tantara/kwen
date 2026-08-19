"""python -m korean_sft {generate,polish,pack,train,pipeline}"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage: python -m korean_sft {generate|polish|pack|train|pipeline|stats|eval} ..."
        )
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "generate":
        from .generate import main as gen

        return gen(rest)
    if cmd == "polish":
        from .polish import main as pol

        return pol(rest)
    if cmd == "pack":
        from .pack import main as pk

        return pk(rest)
    if cmd == "train":
        from .train import main as tr

        return tr(rest)
    if cmd == "stats":
        from .stats import main as st

        return st(rest)
    if cmd == "eval":
        from .eval import main as ev

        return ev(rest)
    if cmd == "pipeline":
        from .diversity import FIVEPAGE_COUNT, ONEPAGE_COUNT, TARGET_COUNT
        from .generate import generate_corpus
        from .pack import pack_corpus
        from .paths import (
            FIVEPAGE_POLISHED_PATH,
            FIVEPAGE_RAW_PATH,
            FIVEPAGE_SFT_PATH,
            ONEPAGE_POLISHED_PATH,
            ONEPAGE_RAW_PATH,
            ONEPAGE_SFT_PATH,
            POLISHED_PATH,
            RAW_PATH,
            SFT_PATH,
        )
        from .polish import polish_corpus

        length = "short"
        if "--length" in rest:
            i = rest.index("--length")
            if i + 1 < len(rest):
                length = rest[i + 1]
        if length == "halfpage":
            length = "onepage"
        if length == "onepage":
            generate_corpus(path=ONEPAGE_RAW_PATH, count=ONEPAGE_COUNT, form="onepage")
            polish_corpus(src=ONEPAGE_RAW_PATH, dst=ONEPAGE_POLISHED_PATH)
            pack_corpus(src=ONEPAGE_POLISHED_PATH, dst=ONEPAGE_SFT_PATH)
        elif length == "fivepage":
            generate_corpus(path=FIVEPAGE_RAW_PATH, count=FIVEPAGE_COUNT, form="fivepage")
            polish_corpus(src=FIVEPAGE_RAW_PATH, dst=FIVEPAGE_POLISHED_PATH)
            pack_corpus(src=FIVEPAGE_POLISHED_PATH, dst=FIVEPAGE_SFT_PATH)
        else:
            generate_corpus(path=RAW_PATH, count=TARGET_COUNT, form="short")
            polish_corpus(src=RAW_PATH, dst=POLISHED_PATH)
            pack_corpus(src=POLISHED_PATH, dst=SFT_PATH)
        print("pipeline done")
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
