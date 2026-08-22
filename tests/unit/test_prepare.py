"""data.prepare against a tiny fixture shaped exactly like the real PaySim CSV
-- the real 470MB file cannot be vendored here (size and licence), but its
column layout is public and stable, so a handful of rows in that exact layout
is enough to test the ingest path without the file.
"""

import os

import numpy as np
import pytest

from data.prepare import load
from data.synth import TYPES

CSV = """step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
1,PAYMENT,9839.64,C1231006815,170136.0,160296.36,M1979787155,0.0,0.0,0,0
1,TRANSFER,181.0,C1305486145,181.0,0.0,C553264065,0.0,0.0,1,0
1,CASH_OUT,181.0,C840083671,181.0,0.0,C38997010,21182.0,0.0,1,0
2,DEBIT,5337.77,C712410124,41720.0,36382.23,C195600860,41898.0,40348.79,0,0
2,CASH_IN,9644.94,C1900366749,4465.0,0.0,C997608398,10845.0,157982.12,0,0
"""


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "paysim_fixture.csv"
    p.write_text(CSV)
    return str(p)


def test_schema_matches_data_synth_exactly(csv_path):
    """Everything downstream -- engine.train.build, engine.rebuild,
    engine.gbdt -- reads these exact keys. A mismatch here would fail far from
    its cause, deep inside sharding."""
    from data.synth import generate
    real = load(csv_path)
    synthetic = generate(n_subjects=5, seed=0)
    assert set(real) == set(synthetic)


def test_row_count_and_values_are_exact(csv_path):
    d = load(csv_path)
    assert len(d["step"]) == 5
    assert d["nameOrig"][0] == "C1231006815"
    assert d["amount"][0] == pytest.approx(9839.64)
    assert d["isFraud"].tolist() == [0, 1, 1, 0, 0]


def test_type_strings_map_to_the_same_indices_data_synth_uses(csv_path):
    d = load(csv_path)
    # Row order in the fixture: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN.
    expected = [TYPES.index(t) for t in ("PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN")]
    assert d["type_idx"].tolist() == expected


def test_unrecognised_type_raises_rather_than_silently_dropping():
    """A schema drift in a future PaySim export (a new transaction type) must
    surface immediately, not disappear into a KeyError three modules away or,
    worse, get silently coerced into the wrong bucket."""
    import tempfile
    bad = CSV.replace("PAYMENT", "SOMETHING_NEW", 1)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(bad)
        path = f.name
    with pytest.raises(KeyError):
        load(path)


def test_can_feed_straight_into_engine_train_build(tmp_path, monkeypatch, csv_path):
    """The actual promise: swapping data.synth.generate() for real rows needs
    no other change. Runs the real build() with loaded (fixture) data instead
    of the synthetic generator and confirms shards and routing come out.

    Does NOT override NUM_SHARDS via monkeypatch: engine.sharder imports it as
    its own separate binding from config.settings (every module that imports a
    settings constant gets its own frozen copy at import time), so patching
    engine.train's copy alone silently does not change which shard a subject
    lands in -- this is exactly how NUM_SHARDS at the real default (5) surfaced
    the empty-shard bug engine/slicer.py just got a guard for: 5 subjects over
    5 shards leaves several shards empty by construction, no monkeypatch
    needed to trigger it.
    """
    from config.settings import NUM_SHARDS
    from engine import train as train_mod

    monkeypatch.setattr(train_mod, "SHARD_DIR", str(tmp_path / "shards"))

    raw = load(csv_path)
    routing = train_mod.build(raw=raw)

    assert len(routing) == 5  # 5 distinct nameOrig in the fixture
    total_rows = sum(len(train_mod.load_shard(k)["step"])
                     for k in range(NUM_SHARDS) if os.path.exists(train_mod.shard_path(k)))
    assert total_rows == 5
