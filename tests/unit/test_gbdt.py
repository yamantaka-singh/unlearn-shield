"""SISA over gradient-boosted trees (engine/gbdt.py).

The claims here are stronger than the MLP path's, and cheaper to check,
because tree rollback is truncation rather than a checkpoint reload.
"""

import json

import numpy as np
import pytest
import xgboost as xgb

from config.settings import NUM_SLICES
from data.synth import generate
from engine import gbdt


def corpus(n_subjects=120, seed=5):
    """A shard-shaped corpus: subject-aligned slices, as engine/slicer.py
    produces (ADR 0005). Every one of a subject's rows shares a slice."""
    rng = np.random.default_rng(seed)
    records = generate(n_subjects=n_subjects, seed=seed)
    subjects, inverse = np.unique(records["nameOrig"], return_inverse=True)
    slice_of_subject = rng.integers(0, NUM_SLICES, size=len(subjects))
    records["slice_idx"] = slice_of_subject[inverse]
    records["subject_ref"] = records["nameOrig"]
    return records, subjects, slice_of_subject


def digest(booster: xgb.Booster) -> str:
    return booster.save_raw(raw_format="json").hex()


def test_truncation_equals_training_fewer_rounds():
    """The load-bearing claim of this whole engine.

    If `booster[0:n]` is not exactly what training n rounds would have
    produced, then rollback is an approximation and the unlearning guarantee
    is not exact.
    """
    records, _, _ = corpus()
    rows = np.arange(len(records["step"]))
    d = xgb.DMatrix(gbdt.features(records, rows), label=records["isFraud"][rows])

    long_run = xgb.train(gbdt.PARAMS, d, num_boost_round=30)
    short_run = xgb.train(gbdt.PARAMS, d, num_boost_round=12)

    np.testing.assert_array_equal(long_run[0:12].predict(d), short_run.predict(d))


def test_rebuild_equals_a_clean_retrain_on_retained_data():
    """The exact-unlearning property, stated as strongly as it can be.

    A rebuild keeps the trees from slices before the target's, then boosts
    forward on purged data. Because slices are subject-aligned, those kept
    trees never saw the target -- so they are exactly the trees a from-scratch
    retrain on retained data would have produced for those same rounds.

    Byte-identical, not merely close. "The model behaves similarly" is what
    gradient-ascent unlearning offers, and the thing this project exists to
    improve on.
    """
    records, subjects, slice_of_subject = corpus()
    target_idx = int(np.flatnonzero(slice_of_subject > 0)[0])
    target = subjects[target_idx]
    min_slice = int(slice_of_subject[target_idx])

    full = gbdt.train_shard(0, records)
    rebuilt, retained = gbdt.rebuild_booster(0, [target], records, min_slice, full)

    from_scratch = gbdt.train_shard(0, retained)

    assert digest(rebuilt) == digest(from_scratch)


def test_rebuild_from_slice_zero_also_matches_a_clean_retrain():
    """The expensive path: nothing to keep, so it retrains the whole shard.
    It must still land in exactly the same place."""
    records, subjects, slice_of_subject = corpus(seed=9)
    target_idx = int(np.flatnonzero(slice_of_subject == 0)[0])
    target = subjects[target_idx]

    full = gbdt.train_shard(0, records)
    rebuilt, retained = gbdt.rebuild_booster(0, [target], records, 0, full)

    assert digest(rebuilt) == digest(gbdt.train_shard(0, retained))


def test_rollback_to_slice_zero_keeps_nothing():
    records, _, _ = corpus()
    full = gbdt.train_shard(0, records)
    assert gbdt.rollback(full, 0) is None


def test_rollback_keeps_exactly_the_earlier_slices_trees():
    records, _, _ = corpus()
    full = gbdt.train_shard(0, records, trees_per_slice=6)
    for to_slice in range(1, NUM_SLICES):
        assert gbdt.rollback(full, to_slice, trees_per_slice=6).num_boosted_rounds() == to_slice * 6


def test_training_is_deterministic():
    """Same reason as ADR 0003: a manifest is only auditable if a rebuild is
    reproducible, so the tree engine has to hold the same property the MLP
    engine does."""
    records, _, _ = corpus(seed=13)
    assert digest(gbdt.train_shard(0, records)) == digest(gbdt.train_shard(0, records))


def test_base_score_is_a_constant_not_a_fitted_statistic():
    """base_score must not be fitted to labels, since a rebuild continuing
    from a truncated booster inherits it rather than recomputing -- the shape
    of ADR 0004's preprocessing leak.

    Asserted as a config property, not behaviourally, and that is a deliberate
    downgrade from an earlier draft of this test. The behavioural version does
    not reproduce: XGBoost estimates base_score on the first train() call,
    which here sees slice 0 only, so a slice->=1 subject is absent from the
    estimate and a slice-0 subject forces a fresh one. Both paths are clean
    already. A behavioural assertion would therefore pass with the pin removed
    -- a test that looks like a guard and guards nothing.

    Pinning is defence against that safety being incidental (it depends on
    when XGBoost happens to estimate) rather than designed. This test guards
    the pin; engine/gbdt.py explains why the pin is worth having.
    """
    assert gbdt.PARAMS.get("base_score") == 0.5, (
        "base_score must stay a constant -- unpinned it is fitted to labels, "
        "and a rebuild inherits the fitted value from the pre-purge model")

    # And confirm the pin actually reaches the model, rather than being a
    # dict key XGBoost silently ignores.
    records, _, _ = corpus(seed=17)
    trained = gbdt.train_shard(0, records)
    base = json.loads(trained.save_config())["learner"]["learner_model_param"]["base_score"]
    assert float(base) == 0.5


def test_purged_subject_is_absent_from_the_retained_rows():
    records, subjects, slice_of_subject = corpus(seed=19)
    target = subjects[int(np.flatnonzero(slice_of_subject > 0)[0])]
    assert (records["subject_ref"] == target).sum() > 0

    _, retained = gbdt.rebuild_booster(0, [target], records,
                               int(slice_of_subject[np.flatnonzero(subjects == target)[0]]),
                               gbdt.train_shard(0, records))

    assert (retained["subject_ref"] == target).sum() == 0


def test_purging_an_absent_subject_is_refused():
    records, _, _ = corpus()
    with pytest.raises(ValueError, match="none of"):
        gbdt.rebuild_booster(0, ["not-a-real-subject"], records, 1, gbdt.train_shard(0, records))


def test_resuming_without_a_booster_is_refused():
    records, _, _ = corpus()
    with pytest.raises(ValueError, match="needs a booster"):
        gbdt.train_shard(0, records, from_slice=2, booster=None)


def test_features_of_one_row_do_not_depend_on_any_other_row():
    """Exactly the property a fitted scaler breaks.

    ADR 0004's leak is that a scaler fit across rows bakes every subject's
    values into every other subject's features. Here, changing rows 50+
    leaves rows 0-9 bit-identical -- there is no cross-row statistic to
    carry anyone's data anywhere.
    """
    a, _, _ = corpus(seed=1)
    b = {k: v.copy() for k, v in a.items()}
    b["amount"][50:] *= 1000.0

    rows = np.arange(10)
    np.testing.assert_array_equal(gbdt.features(a, rows), gbdt.features(b, rows))


# --------------------------------------------------------------------------
# rebuild_batch_by_ref: the manifest-emitting entrypoint, against real files.
# Mirrors tests/unit/test_rebuild.py's `built` fixture pattern.
# --------------------------------------------------------------------------

from engine import train as train_mod


@pytest.fixture
def gbdt_corpus(tmp_path, monkeypatch):
    shard_dir = str(tmp_path / "shards")
    monkeypatch.setattr(train_mod, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(gbdt, "SHARD_DIR", shard_dir)
    routing = train_mod.build(n_subjects=150, seed=41)
    gbdt.build(routing)
    return routing


def _target_id(routing, n_subjects=150):
    from config.settings import subject_ref
    return next(s for s in (f"C{i:07d}" for i in range(n_subjects)) if subject_ref(s) in routing)


def test_rebuild_emits_a_certificate_the_standalone_verifier_accepts(gbdt_corpus, monkeypatch):
    from nacl.signing import SigningKey
    from verify.verifier_cli import verify_certificate

    key = SigningKey.generate()
    monkeypatch.setenv("UNLEARNSHIELD_SIGNING_KEY", bytes(key).hex())

    target = _target_id(gbdt_corpus)
    result = gbdt.rebuild(target)

    # key.verify_key directly -- not a monkeypatched load_public_key -- since
    # a name imported before a monkeypatch stays bound to the original
    # function object; the patch only changes what a FRESH lookup of the
    # module attribute sees.
    ok, findings = verify_certificate(dict(result["manifest"]), key.verify_key)
    assert ok, findings
    assert "gbdt-shard" in result["manifest"]["model_version"]


def test_rebuild_purges_the_subject_from_the_shard_file(gbdt_corpus, monkeypatch):
    from nacl.signing import SigningKey
    monkeypatch.setenv("UNLEARNSHIELD_SIGNING_KEY", bytes(SigningKey.generate()).hex())

    target = _target_id(gbdt_corpus)
    from config.settings import subject_ref
    ref = subject_ref(target)
    shard = gbdt_corpus[ref]["shard"]

    before = train_mod.load_shard(shard)
    assert (before["subject_ref"] == ref).sum() > 0

    gbdt.rebuild(target)

    after = train_mod.load_shard(shard)
    assert (after["subject_ref"] == ref).sum() == 0


def test_rebuild_removes_the_routing_entry(gbdt_corpus, monkeypatch):
    from nacl.signing import SigningKey
    monkeypatch.setenv("UNLEARNSHIELD_SIGNING_KEY", bytes(SigningKey.generate()).hex())

    target = _target_id(gbdt_corpus)
    from config.settings import subject_ref
    ref = subject_ref(target)

    gbdt.rebuild(target)

    assert ref not in train_mod.load_routing()


def test_repeated_erasure_of_the_same_subject_is_refused(gbdt_corpus, monkeypatch):
    """Not idempotent by itself -- Phase 4's Idempotency-Key is what makes a
    repeat HTTP request safe. Calling this twice directly must fail loudly,
    not silently succeed against a subject who is already gone."""
    from nacl.signing import SigningKey
    monkeypatch.setenv("UNLEARNSHIELD_SIGNING_KEY", bytes(SigningKey.generate()).hex())

    target = _target_id(gbdt_corpus)
    gbdt.rebuild(target)
    with pytest.raises(KeyError):
        gbdt.rebuild(target)


def test_config_digest_changes_when_trees_per_slice_changes(monkeypatch):
    before = gbdt.config_digest()
    monkeypatch.setattr(gbdt, "TREES_PER_SLICE", gbdt.TREES_PER_SLICE + 1)
    assert gbdt.config_digest() != before
