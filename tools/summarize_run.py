#!/usr/bin/env python3
"""Validate a completed CMHA experiment and optionally write compact summaries."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmha_results import summarize_run, write_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = summarize_run(args.run_dir) if args.check_only else write_summary(args.run_dir)
    print(json.dumps(result, indent=2))
