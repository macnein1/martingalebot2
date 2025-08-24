import sqlite3
from martingale_lab.storage.experiments_store import ExperimentsStore


def test_upsert_inserts_rows_memory_db():
    # Use a shared in-memory DB URI so multiple connections see the same DB
    db_uri = "file:memdb1?mode=memory&cache=shared"
    store = ExperimentsStore(db_uri)

    exp_id = store.create_experiment("unit", {"notes": "unit"}, run_id="RUN-UNIT")

    items = [
        {
            "stable_id": f"sid-{i}",
            "score": float(i),
            "payload": {"overlap": 10 + i, "orders": 5 + i},
            "sanity": {},
            "diagnostics": {},
            "penalties": {},
        }
        for i in range(10)
    ]

    affected = store.upsert_results(exp_id, items)

    # Rows should be inserted
    assert affected >= 1

    # Verify count using a fresh connection to the same memory DB via URI
    con = sqlite3.connect(db_uri, uri=True)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM results WHERE experiment_id=?", (exp_id,))
    n = cur.fetchone()[0]
    assert n >= 10
    con.close() 
