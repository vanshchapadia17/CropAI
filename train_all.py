"""Run both training pipelines (crop classifier + disease classifier) in sequence.

Usage from the project root:
    virtualEN/python.exe train_all.py

Each pipeline is independent — if one fails, the other still runs. A summary
is printed at the end, and the script exits non-zero if anything failed.
"""

import sys
import time
from datetime import datetime

from src.logger import logging
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.disease_train_pipeline import DiseaseTrainPipeline


def _banner(text: str) -> None:
    bar = "=" * 70
    line = f"  {text}"
    print(f"\n{bar}\n{line}\n{bar}", flush=True)
    logging.info(text)


def _run(name: str, pipeline) -> tuple:
    """Run one pipeline; return (name, success, duration_seconds, error)."""
    _banner(f"STARTING: {name}  ({datetime.now().strftime('%H:%M:%S')})")
    start = time.time()
    try:
        pipeline.run_pipeline()
        elapsed = time.time() - start
        _banner(f"DONE: {name}  ({elapsed/60:.1f} min)")
        return name, True, elapsed, None
    except Exception as e:
        elapsed = time.time() - start
        err = f"{type(e).__name__}: {e}"
        _banner(f"FAILED: {name}  ({elapsed/60:.1f} min) — {err}")
        logging.exception(f"{name} failed")
        return name, False, elapsed, err


def main() -> int:
    overall_start = time.time()
    results = [
        _run("Crop Classifier Pipeline", TrainPipeline()),
        _run("Disease Classifier Pipeline", DiseaseTrainPipeline()),
    ]

    _banner("SUMMARY")
    for name, ok, secs, err in results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}]  {name:<32}  {secs/60:>5.1f} min"
              + (f"  -> {err}" if err else ""))
    total_min = (time.time() - overall_start) / 60
    print(f"\n  Total: {total_min:.1f} min")

    return 0 if all(ok for _, ok, _, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
