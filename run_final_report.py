"""Compatibility entry point for the final model report."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-final-report")

import traceback
from gmsc.final_report import main

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n=== run_final_report.py Error ===")
        print(traceback.format_exc())
        raise
