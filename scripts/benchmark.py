"""What SISA actually costs, measured on real PaySim and written to JSON.

    python -m scripts.benchmark --csv /path/to/paysim.csv --rows 400000 \
        --out bench.json

ADR 0012 reported per-shard IN-SAMPLE ROC-AUC. Both halves of that are weak,
and this script exists because they were quoted as if they were not:

  * In-sample. Scoring a booster on the rows it just trained on measures
    memorisation. The only number worth publishing is on data the model has
    not seen, so this splits TEMPORALLY -- train on the earlier steps, test on
    the later ones. A random split would leak: PaySim's fraud arrives in
    bursts within a step, so a randomly held-out row often has its own burst
    in the training half.

  * ROC-AUC, at 0.1% prevalence. It is dominated by the negative class and
    stays near 1.0 for models that are useless in production. Average
    precision and precision-at-recall are reported alongside it, and they are
    the ones to read (data/eval_set.py explains the difference).

The comparison that answers "what does sharding cost me" is `sisa` against
`unsharded`: the same rows, the same hyperparameters, the same total boosting
rounds -- the ONLY difference is that one is split into NUM_SHARDS isolated
models whose votes get averaged, and the other is one model over everything.
Any gap between them is the price of being able to erase a subject.

Environment is set before the first import on purpose. Every module binds
SHARD_DIR at import time (tests/conftest.py::override_shard_dir explains what
goes wrong otherwise), and setting it here is the same thing production does:
fix the config, then start the process.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time


def _configure(scratch: str) -> None:
    """Point every path at a scratch directory BEFORE config.settings loads."""
    os.environ["SHARD_DIR"] = os.path.join(scratch, "shards")
    os.environ["CHECKPOINT_DIR"] = os.path.join(scratch, "ckpt")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    if "config.settings" in sys.modules:
        raise RuntimeError(
            "config.settings was imported before _configure ran -- the scratch "
            "directories would be ignored and this benchmark would train "
            "against the repo's real data/shards/. Run this as a script.")


def _metrics(y_true, y_score) -> dict:
    from data.eval_set import auc, average_precision, precision_at_recall
    return {
        "roc_auc": auc(y_true, y_score),
        "average_precision": average_precision(y_true, y_score),
        "precision_at_recall_50": precision_at_recall(y_true, y_score, 0.50),
        "precision_at_recall_80": precision_at_recall(y_true, y_score, 0.80),
    }


def run(csv_path: str, rows: int, train_fraction: float, seed: int) -> dict:
    import numpy as np
    import xgboost as xgb

    from config.settings import NUM_SHARDS, NUM_SLICES
    from data.prepare import load as load_paysim
    from engine import gbdt
    from engine import train as train_mod

    raw = load_paysim(csv_path) if rows <= 0 else load_paysim(csv_path, max_rows=rows)
    n = len(raw["step"])

    # Temporal split. `step` is PaySim's hour counter, so this is "train on
    # the past, score the future" -- the split a deployed fraud model lives
    # under, and the one that cannot leak a burst across the boundary.
    cut = float(np.quantile(raw["step"], train_fraction))
    is_train = raw["step"] <= cut
    train_raw = {k: v[is_train] for k, v in raw.items()}
    test_raw = {k: v[~is_train] for k, v in raw.items()}
    if test_raw["isFraud"].sum() == 0:
        raise SystemExit(f"no fraud in the held-out half (step > {cut}); use more rows")

    result = {
        "dataset": {
            "source": os.path.basename(csv_path),
            "rows_total": int(n),
            "rows_train": int(is_train.sum()),
            "rows_test": int((~is_train).sum()),
            "split": f"temporal, step <= {cut:g}",
            "fraud_train": int(train_raw["isFraud"].sum()),
            "fraud_test": int(test_raw["isFraud"].sum()),
            "prevalence_test": float(test_raw["isFraud"].mean()),
        },
        "config": {"num_shards": NUM_SHARDS, "num_slices": NUM_SLICES,
                   "trees_per_slice": gbdt.TREES_PER_SLICE, "seed": seed},
    }

    routing = train_mod.build(raw=train_raw, seed=seed)
    result["dataset"]["subjects_train"] = len(routing)

    test_rows = np.arange(len(test_raw["step"]))
    y_test = test_raw["isFraud"]

    # --- SISA ensemble: NUM_SHARDS isolated boosters, votes averaged --------
    t0 = time.perf_counter()
    gbdt.build(routing)
    sisa_train_seconds = time.perf_counter() - t0

    shards = sorted({e["shard"] for e in routing.values()})
    per_shard = {}
    shard_scores = []
    for shard in shards:
        booster = gbdt.load_booster(shard)
        scores = gbdt.predict(booster, test_raw, test_rows)
        shard_scores.append(scores)
        per_shard[str(shard)] = {
            "rows": int(len(train_mod.load_shard(shard)["step"])),
            **_metrics(y_test, scores),
        }
    result["gbdt_sisa"] = {
        "train_seconds": sisa_train_seconds,
        "ensemble": _metrics(y_test, np.mean(shard_scores, axis=0)),
        "per_shard": per_shard,
    }

    # --- Unsharded baseline: one booster, same rows, same total rounds ------
    # Matched deliberately. A baseline trained for a different number of
    # rounds would confound "sharding costs accuracy" with "one model got more
    # boosting", and that confound is exactly how an architecture gets blamed
    # for a hyperparameter.
    all_rows = np.arange(len(train_raw["step"]))
    features = gbdt.features(train_raw, all_rows)
    dtrain = xgb.DMatrix(features, label=train_raw["isFraud"])
    total_rounds = gbdt.TREES_PER_SLICE * NUM_SLICES
    t0 = time.perf_counter()
    flat = xgb.train(gbdt.PARAMS, dtrain, num_boost_round=total_rounds)
    unsharded_seconds = time.perf_counter() - t0
    flat_scores = flat.predict(xgb.DMatrix(gbdt.features(test_raw, test_rows)))
    result["gbdt_unsharded"] = {
        "train_seconds": unsharded_seconds,
        "rounds": total_rounds,
        **_metrics(y_test, flat_scores),
    }

    # --- What an erasure costs, broken down by phase -------------------------
    # Reported per phase rather than as one wall-clock number, because the one
    # number is actively misleading at this scale. SISA's claim is about
    # RETRAINING: roll back to the rollback point and boost forward instead of
    # retraining the shard from scratch. That is the `retrain_from_rollback` vs
    # `retrain_from_scratch` pair below, and it is the only pair that tests the
    # claim.
    #
    # End-to-end erasure also builds a sparse Merkle root over every retained
    # subject and rewrites routing.json, and at hundreds of thousands of
    # subjects those dominate everything else -- the first version of this
    # script timed only the total and made SISA look ~70x SLOWER than a full
    # retrain, which is what a mismatched comparison buys you. Separating the
    # phases is what makes the number mean something, and the proof cost is
    # worth seeing on its own regardless: it is the part that scales with
    # corpus size rather than with shard size.
    from verify.smt import build_root

    target = max(routing, key=lambda r: routing[r]["record_count"])
    entry = routing[target]
    shard, min_slice = entry["shard"], entry["min_slice_idx"]

    records = train_mod.load_shard(shard)
    booster = gbdt.load_booster(shard)
    keep = records["subject_ref"] != target
    retained = {k: v[keep] for k, v in records.items()}

    t0 = time.perf_counter()
    gbdt.train_shard(shard, retained, from_slice=min_slice,
                     booster=gbdt.rollback(booster, min_slice))
    retrain_from_rollback = time.perf_counter() - t0

    t0 = time.perf_counter()
    gbdt.train_shard(shard, retained, from_slice=0, booster=None)
    retrain_from_scratch = time.perf_counter() - t0

    # The cost of erasing ONE subject tells you almost nothing, because the
    # saving is entirely determined by which slice that subject sits in --
    # rolling back to slice 4 reboosts one slice, rolling back to slice 0 is a
    # full retrain by definition. Timing a single arbitrary subject reports
    # wherever that subject happened to land. So: the cost at every rollback
    # point, alongside how many subjects actually sit at each one, which is
    # what turns the curve into an expected cost for this corpus.
    #
    # ADR 0005's design intent is visible in that distribution rather than in
    # any single timing: churn-ascending ordering is supposed to concentrate
    # likely-deleted subjects in the HIGH slices, where rollback is cheap.
    by_slice = {}
    for s in range(NUM_SLICES):
        t0 = time.perf_counter()
        gbdt.train_shard(shard, retained, from_slice=s,
                         booster=gbdt.rollback(booster, s))
        seconds = time.perf_counter() - t0
        population = sum(1 for e in routing.values() if e["min_slice_idx"] == s)
        by_slice[str(s)] = {
            "retrain_seconds": seconds,
            "speedup_vs_scratch": retrain_from_scratch / seconds,
            "subjects_at_this_rollback_point": population,
            "share_of_subjects": population / len(routing),
        }
    expected_retrain = sum(v["retrain_seconds"] * v["share_of_subjects"]
                           for v in by_slice.values())

    retained_refs = set(retained["subject_ref"].tolist())
    t0 = time.perf_counter()
    build_root(retained_refs)
    proof_seconds = time.perf_counter() - t0

    # Last, because it mutates: pops the routing row and rewrites the shard.
    t0 = time.perf_counter()
    erase = gbdt.rebuild_batch_by_ref([target], sign=False)
    end_to_end = time.perf_counter() - t0

    result["erasure"] = {
        "shard": shard,
        "min_slice_idx": min_slice,
        "record_count": entry["record_count"],
        "resumed_from": erase["resumed_from"],
        "slices_retrained": erase["slices_retrained"],
        "rows_purged": erase["rows_purged"],
        "retrain_from_rollback_seconds": retrain_from_rollback,
        "retrain_from_scratch_seconds": retrain_from_scratch,
        "retrain_speedup": retrain_from_scratch / retrain_from_rollback,
        "by_rollback_point": by_slice,
        "expected_retrain_seconds": expected_retrain,
        "expected_retrain_speedup": retrain_from_scratch / expected_retrain,
        "full_corpus_retrain_seconds": unsharded_seconds,
        "retrain_speedup_vs_full_corpus": unsharded_seconds / retrain_from_rollback,
        "absence_proof_seconds": proof_seconds,
        "subjects_in_proof": len(retained_refs),
        "end_to_end_seconds": end_to_end,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="real PaySim CSV")
    parser.add_argument("--rows", type=int, default=400_000, help="0 for all")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default="benchmark.json")
    args = parser.parse_args()

    scratch = tempfile.mkdtemp(prefix="unlearnshield-bench-")
    try:
        _configure(scratch)
        result = run(args.csv, args.rows, args.train_fraction, args.seed)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    result["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    d, s, u, e = (result["dataset"], result["gbdt_sisa"]["ensemble"],
                  result["gbdt_unsharded"], result["erasure"])
    print(f"{d['rows_train']:,} train / {d['rows_test']:,} test rows, "
          f"{d['fraud_test']} fraud held out ({d['prevalence_test']*100:.3f}%)")
    print(f"{'':22} {'ROC-AUC':>9} {'AP':>9} {'P@R=0.80':>9}")
    for name, m in (("SISA ensemble", s), ("unsharded baseline", u)):
        print(f"{name:22} {m['roc_auc']:9.4f} {m['average_precision']:9.4f} "
              f"{m['precision_at_recall_80']['precision']:9.4f}")
    print(f"erasure retrain by rollback point (from-scratch = "
          f"{e['retrain_from_scratch_seconds']:.2f}s):")
    for s, v in sorted(e["by_rollback_point"].items()):
        print(f"  slice {s}: {v['retrain_seconds']:5.2f}s  "
              f"{v['speedup_vs_scratch']:4.1f}x  "
              f"{v['share_of_subjects']*100:5.1f}% of subjects")
    print(f"  expected: {e['expected_retrain_seconds']:.2f}s "
          f"({e['expected_retrain_speedup']:.1f}x)")
    print(f"absence proof over {e['subjects_in_proof']:,} subjects: "
          f"{e['absence_proof_seconds']:.2f}s "
          f"({e['absence_proof_seconds']/e['end_to_end_seconds']*100:.0f}% of end-to-end "
          f"{e['end_to_end_seconds']:.2f}s)")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
