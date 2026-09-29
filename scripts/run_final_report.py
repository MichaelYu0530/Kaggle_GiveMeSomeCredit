"""Run the frozen final report from any working directory."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-final-report")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmsc.final_report import main

if __name__ == "__main__":
    os.chdir(ROOT)
    main()
