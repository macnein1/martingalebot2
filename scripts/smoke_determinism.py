import subprocess
import sqlite3
import sys
from pathlib import Path

DB = Path("db_results/experiments.db")


def run_once(note: str) -> None:
    subprocess.check_call([
        sys.executable, "-m", "martingale_lab.cli.optimize",
        "--overlap-min", "15", "--overlap-max", "25",
        "--orders-min", "5", "--orders-max", "10",
        "--batches", "5", "--batch-size", "100", "--workers", "2",
        "--db", str(DB), "--seed", "42", "--disable-early-stop",
        "--log-eval-sample", "0.0", "--log-every-batch", "5",
        "--notes", note
    ])


def best_from_db(exp_id: int) -> float:
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT MIN(score) FROM results WHERE experiment_id=?", (exp_id,))
    row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else float("inf")


def main() -> int:
    run_once("det-run-1")
    run_once("det-run-2")
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT id FROM experiments ORDER BY id DESC LIMIT 2;")
    last_two = [r[0] for r in cur.fetchall()][::-1]
    if len(last_two) < 2:
        print("Not enough experiments to compare", file=sys.stderr)
        return 1
    b1, b2 = best_from_db(last_two[0]), best_from_db(last_two[1])
    print("Best1=", b1, "Best2=", b2)
    return 0 if abs(b1 - b2) < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())