"""Real PaySim ingest. Produces the exact same dict schema data/synth.py does,
so nothing downstream (engine/, gbdt.py, tests) needs to know which one ran.

    python -m data.prepare /path/to/PS_*.csv --out data/paysim.npz

The real file is ~470MB / 6.3M rows and is not vendored here, per the licence
on the upstream dataset -- point this at your own local copy.
"""

import argparse
import csv
import sys

import numpy as np

from data.synth import TYPES


def load(path: str) -> dict:
    """Streams the CSV once rather than going through pandas -- one pass, no
    intermediate DataFrame, and no new dependency for a job this stdlib-shaped.
    """
    type_index = {t: i for i, t in enumerate(TYPES)}
    cols = {k: [] for k in ("nameOrig", "step", "type_idx", "amount",
                            "oldbalanceOrg", "newbalanceOrig",
                            "oldbalanceDest", "newbalanceDest", "isFraud")}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cols["nameOrig"].append(row["nameOrig"])
            cols["step"].append(float(row["step"]))
            cols["type_idx"].append(type_index[row["type"]])
            cols["amount"].append(float(row["amount"]))
            cols["oldbalanceOrg"].append(float(row["oldbalanceOrg"]))
            cols["newbalanceOrig"].append(float(row["newbalanceOrig"]))
            cols["oldbalanceDest"].append(float(row["oldbalanceDest"]))
            cols["newbalanceDest"].append(float(row["newbalanceDest"]))
            cols["isFraud"].append(int(row["isFraud"]))

    return {
        "nameOrig": np.array(cols["nameOrig"]),
        "step": np.array(cols["step"], dtype=np.float64),
        "type_idx": np.array(cols["type_idx"], dtype=np.int64),
        "amount": np.array(cols["amount"], dtype=np.float64),
        "oldbalanceOrg": np.array(cols["oldbalanceOrg"], dtype=np.float64),
        "newbalanceOrig": np.array(cols["newbalanceOrig"], dtype=np.float64),
        "oldbalanceDest": np.array(cols["oldbalanceDest"], dtype=np.float64),
        "newbalanceDest": np.array(cols["newbalanceDest"], dtype=np.float64),
        "isFraud": np.array(cols["isFraud"], dtype=np.int64),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--out", default="data/paysim.npz")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="subsample for a fast smoke run; omit for the full file")
    args = parser.parse_args()

    records = load(args.csv_path)
    if args.max_rows and len(records["step"]) > args.max_rows:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(records["step"]), args.max_rows, replace=False))
        records = {k: v[idx] for k, v in records.items()}

    np.savez(args.out, **records)
    n = len(records["step"])
    n_subj = len(set(records["nameOrig"].tolist()))
    print(f"{n} rows, {n_subj} distinct nameOrig, "
          f"fraud rate {records['isFraud'].mean():.5f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
