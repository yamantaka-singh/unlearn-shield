"""Does PaySim actually look like data/synth.py assumes?

    python -m scripts.check_paysim_structure /path/to/PS_*.csv

ADR 0005's whole argument for subject-aligned slicing is that a subject owns
MANY records, so record-level slicing scatters them across slices and the
rollback point collapses to slice 0. data/synth.py bakes in a geometric
record-count distribution to make that visible -- but nobody has checked
whether the real, intended dataset (PaySim) actually has that shape at all.

If real subjects turn out to own ~1 record each, the leak record-level
slicing causes doesn't arise in the first place, and the docstring's own
claim ("equal-sized subjects would hide the imbalance") would be describing
a problem the real data doesn't have.
"""

import argparse
import csv
from collections import Counter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()

    orig_counts, dest_counts = Counter(), Counter()
    n = 0
    with open(args.csv_path, newline="") as f:
        for row in csv.DictReader(f):
            orig_counts[row["nameOrig"]] += 1
            dest_counts[row["nameDest"]] += 1
            n += 1

    def summarize(counts, label):
        vals = sorted(counts.values())
        n_ids = len(vals)
        one = sum(1 for v in vals if v == 1)
        print(f"\n{label}: {n_ids} distinct ids over {n} rows")
        print(f"  exactly 1 record : {one} ({one/n_ids*100:.1f}% of ids)")
        print(f"  median records/id: {vals[n_ids//2]}")
        print(f"  max records/id   : {vals[-1]}")
        print(f"  p99 records/id   : {vals[int(n_ids*0.99)]}")

    summarize(orig_counts, "nameOrig (the erasure unit -- subject_ref)")
    summarize(dest_counts, "nameDest (counterparty, not the erasure unit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
