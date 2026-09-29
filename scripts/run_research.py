"""Run the archived full research workflow from the repository root."""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    os.chdir(ROOT)
    runpy.run_path(str(ROOT / "operation.py"), run_name="__main__")
