#!/usr/bin/env python3
"""Off-policy SFT on frozen-teacher completions sampled once before training.

    .venv/bin/python scripts/run_sft.py configs/sft_<task>.yaml --output runs/<name> [--seed 42]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import main  # noqa: E402

if __name__ == "__main__":
    main("sft")
