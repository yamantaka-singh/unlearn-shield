"""PaySim-shaped synthetic transactions.

The real PaySim CSV is not vendored here (it is ~470MB and not ours to
redistribute). This generator produces the same column contract, so swapping in
real data means replacing `generate` with a CSV read and nothing else:

    step, type, amount, oldbalanceOrg, newbalanceOrig,
    oldbalanceDest, newbalanceDest, isFraud, nameOrig

`nameOrig` is the subject -- the unit a deletion request names, and the reason
slices are subject-aligned rather than record-aligned (see engine/slicer.py).
"""

import numpy as np

# Schema, not a fitted statistic. See engine/preprocessing.py for why declaring
# the enum globally does not violate per-shard isolation.
TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")

NUMERIC_COLUMNS = (
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
)


def generate(n_subjects: int = 800, seed: int = 0, max_step: int = 720) -> dict:
    """Transactions for `n_subjects` subjects, each owning several records.

    Record counts are deliberately uneven (geometric-ish): equal-sized subjects
    would hide the slice-packing imbalance that real data produces.
    """
    rng = np.random.default_rng(seed)
    subject_ids = np.array([f"C{i:07d}" for i in range(n_subjects)])
    counts = 1 + rng.geometric(p=0.25, size=n_subjects)
    total = int(counts.sum())

    owner = np.repeat(subject_ids, counts)
    first_step = np.repeat(rng.integers(0, max_step, size=n_subjects), counts)
    offset = rng.integers(0, 48, size=total)
    step = np.minimum(first_step + offset, max_step).astype(np.float64)

    type_idx = rng.integers(0, len(TYPES), size=total)
    amount = rng.lognormal(mean=7.0, sigma=1.6, size=total)
    old_org = rng.lognormal(mean=8.0, sigma=1.8, size=total)
    new_org = np.maximum(old_org - amount, 0.0)
    old_dest = rng.lognormal(mean=8.0, sigma=1.8, size=total)
    new_dest = old_dest + amount

    # Fraud concentrates in TRANSFER/CASH_OUT that drain the origin account,
    # which is the shape PaySim has and the shape that makes the minority-class
    # degradation in Phase 5 show up at all.
    drains = new_org <= 1e-9
    risky = np.isin(type_idx, [TYPES.index("TRANSFER"), TYPES.index("CASH_OUT")])
    p_fraud = np.where(drains & risky, 0.25, 0.001)
    is_fraud = (rng.random(total) < p_fraud).astype(np.int64)

    return {
        "nameOrig": owner,
        "step": step,
        "type_idx": type_idx.astype(np.int64),
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "isFraud": is_fraud,
    }


if __name__ == "__main__":
    d = generate()
    n = len(d["step"])
    print(f"{n} records, {len(set(d['nameOrig']))} subjects, "
          f"fraud rate {d['isFraud'].mean():.4f}, "
          f"records/subject max {np.bincount(np.unique(d['nameOrig'], return_inverse=True)[1]).max()}")
