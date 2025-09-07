#!/usr/bin/env python3
import sys
import sqlite3
import json
from collections import Counter
from typing import List, Tuple


def fetch_top_schedules(db_path: str, limit: int) -> List[str]:
	con = sqlite3.connect(db_path)
	cur = con.cursor()
	cur.execute(
		"""
		WITH top AS (
		  SELECT schedule_json
		  FROM results r JOIN (SELECT MAX(id) id FROM experiments) l ON r.experiment_id=l.id
		  WHERE r.score > 0
		  ORDER BY r.score ASC
		  LIMIT ?
		)
		SELECT schedule_json FROM top
		""",
		(limit,),
	)
	rows = [r[0] for r in cur.fetchall()]
	con.close()
	return rows


def canonical_key(schedule_json: str) -> Tuple:
	try:
		s = json.loads(schedule_json)
		arr = s.get("volume_pct_norm_2dp") or s.get("volume_pct")
		if not isinstance(arr, list):
			return ("INVALID",)
		# ensure tuple for hashing
		return tuple(round(float(x), 2) for x in arr)
	except Exception:
		return ("INVALID",)


def main():
	if len(sys.argv) < 3:
		print("Usage: report_unique.py <db_path> <top_n>")
		return 1
	db = sys.argv[1]
	top_n = int(sys.argv[2])
	raw = fetch_top_schedules(db, top_n)
	keys = [canonical_key(js) for js in raw]
	cnt = Counter(keys)
	# remove invalid
	if ("INVALID",) in cnt:
		del cnt[("INVALID",)]
	print(f"Unique patterns in top-{top_n}: {len(cnt)}/{top_n}")
	for i, (k, c) in enumerate(cnt.most_common(), 1):
		print(f"{i:3d}. count={c:3d} pattern_head={list(k)[:6]} ...")
	return 0


if __name__ == "__main__":
	sys.exit(main())
