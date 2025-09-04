#!/usr/bin/env python3
import sys
import sqlite3
import json
from typing import Dict, Any, List, Tuple

from martingale_lab.ab_testing.ab_tester import ABTester


def load_unique_strategies(db_path: str, top_n: int) -> List[Dict[str, Any]]:
	con = sqlite3.connect(db_path)
	cur = con.cursor()
	cur.execute(
		"""
		WITH top AS (
		  SELECT payload_json
		  FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
		  WHERE r.score > 0
		  ORDER BY r.score ASC
		  LIMIT ?
		)
		SELECT payload_json FROM top
		""",
		(top_n,),
	)
	rows = [json.loads(r[0]) for r in cur.fetchall()]
	con.close()
	# Extract unique by normalized rounded volumes
	seen = set()
	unique: List[Dict[str, Any]] = []
	for row in rows:
		sch: Dict[str, Any] = row.get("schedule", {})
		v = sch.get("volume_pct_norm_2dp") or sch.get("volume_pct") or []
		key = tuple(round(float(x), 2) for x in v)
		if key and key not in seen:
			seen.add(key)
			unique.append({
				"id": row.get("stable_id", f"idx_{len(unique)}"),
				"volumes": v,
			})
	return unique


def main() -> int:
	if len(sys.argv) < 3:
		print("Usage: ab_unique_compare.py <db_path> <top_n>")
		return 1
	db = sys.argv[1]
	top_n = int(sys.argv[2])
	strategies = load_unique_strategies(db, top_n)
	if len(strategies) < 2:
		print("Need at least 2 unique strategies")
		return 1
	tester = ABTester(db_path="ab_tests.db")
	# Compare first two unique strategies as a simple demo
	res = tester.compare_strategies(strategies[0], strategies[1], market_scenarios=2000, confidence='medium')
	print(f"Compared {strategies[0]['id']} vs {strategies[1]['id']}")
	for k, v in res.items():
		print(f"{k}: winner={v.winner} p={v.p_value:.4f} a_mean={v.a_mean:.4f} b_mean={v.b_mean:.4f}")
	return 0


if __name__ == "__main__":
	sys.exit(main())