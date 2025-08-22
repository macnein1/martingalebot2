"""
Smoke test to ensure end-to-end pipeline writes to DB and logs run.
Run via:
python -c "from martingale_lab.tests.test_real_optimization import run_smoke; run_smoke()"
"""
from __future__ import annotations

import sys
import time
import sqlite3
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_smoke(timeout_s: float = 30.0) -> None:
    root = _project_root()
    sys.path.insert(0, str(root))

    from ui.utils.optimization_bridge import optimization_bridge

    db_path = str(root / "db_results/experiments.db")
    params = dict(min_overlap=15.0, max_overlap=16.0, min_order=3, max_order=4)

    res = optimization_bridge.start_optimization(params, db_path=db_path)
    if not res.get("success"):
        raise RuntimeError(f"Start failed: {res}")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM experiments")
                n = int(cur.fetchone()[0])
                if n >= 1:
                    break
        except Exception:
            pass
        time.sleep(0.5)

    else:
        raise TimeoutError("Smoke test timed out waiting for experiments row")

    # best-effort stop
    try:
        optimization_bridge.stop_optimization()
    except Exception:
        pass

    print("SMOKE OK: experiments row exists and background job started")


if __name__ == "__main__":
    run_smoke()


