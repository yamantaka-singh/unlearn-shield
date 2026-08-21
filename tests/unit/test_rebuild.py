import json
import os

import numpy as np
import pytest
import torch

from config.determinism import state_dict_digest
from config.settings import NUM_SHARDS, NUM_SLICES, subject_ref
from engine import rebuild as rebuild_mod
from engine import train as train_mod


@pytest.fixture
def built(tmp_path, monkeypatch):
    """A small five-shard corpus built and trained in a temp dir."""
    shard_dir, ckpt_dir = str(tmp_path / "shards"), str(tmp_path / "ckpt")
    monkeypatch.setattr(train_mod, "SHARD_DIR", shard_dir)
    monkeypatch.setattr(train_mod, "CHECKPOINT_DIR", ckpt_dir)
    monkeypatch.setattr(rebuild_mod, "SHARD_DIR", shard_dir)
    routing = train_mod.build(n_subjects=200, seed=7)
    for k in range(NUM_SHARDS):
        train_mod.train_shard(k)
    return routing, shard_dir


def test_every_subject_occupies_exactly_one_slice(built):
    """The invariant the rollback point depends on, checked on real shard files.

    If a subject's records span slices, the rollback point is the minimum among
    them, which for a multi-record subject collapses to slice 0 -- a full-shard
    retrain every time.
    """
    _, shard_dir = built
    for k in range(NUM_SHARDS):
        r = train_mod.load_shard(k)
        order = np.argsort(r["subject_ref"])
        refs, slices = r["subject_ref"][order], r["slice_idx"][order]
        boundaries = np.flatnonzero(refs[1:] != refs[:-1])
        groups = np.split(slices, boundaries + 1)
        assert all(len(set(g.tolist())) == 1 for g in groups)


def test_purged_subject_is_absent_from_every_slice(built):
    """Not merely from slices at or after the rollback point.

    Checking only slices >= rollback is what let the original schema's single
    `slice_idx` look correct while leaving a multi-slice subject's earlier rows
    baked into the checkpoint the rebuild resumed from.
    """
    routing, _ = built
    target = next(s for s in (f"C{i:07d}" for i in range(200))
                  if subject_ref(s) in routing)
    ref = subject_ref(target)
    shard = routing[ref]["shard"]

    before = train_mod.load_shard(shard)
    assert (before["subject_ref"] == ref).sum() > 0

    rebuild_mod.rebuild(target)

    after = train_mod.load_shard(shard)
    assert (after["subject_ref"] == ref).sum() == 0
    for slice_idx in range(NUM_SLICES):
        rows = after["slice_idx"] == slice_idx
        assert ref not in set(after["subject_ref"][rows].tolist())


def test_other_subjects_survive_the_rebuild(built):
    routing, _ = built
    target = next(s for s in (f"C{i:07d}" for i in range(200)) if subject_ref(s) in routing)
    shard = routing[subject_ref(target)]["shard"]
    before = train_mod.load_shard(shard)
    survivors = set(before["subject_ref"].tolist()) - {subject_ref(target)}

    rebuild_mod.rebuild(target)

    assert set(train_mod.load_shard(shard)["subject_ref"].tolist()) == survivors


def test_rebuild_changes_the_weights(built):
    routing, _ = built
    target = next(s for s in (f"C{i:07d}" for i in range(200)) if subject_ref(s) in routing)
    shard = routing[subject_ref(target)]["shard"]
    before = state_dict_digest(
        torch.load(train_mod.checkpoint_path(shard, NUM_SLICES - 1), weights_only=True))

    result = rebuild_mod.rebuild(target)

    assert result["result_weights"] != before


def test_rebuild_only_retrains_from_the_rollback_point(built):
    routing, _ = built
    target = next(s for s in (f"C{i:07d}" for i in range(200))
                  if subject_ref(s) in routing and routing[subject_ref(s)]["min_slice_idx"] > 0)
    min_slice = routing[subject_ref(target)]["min_slice_idx"]

    result = rebuild_mod.rebuild(target)

    assert result["slices_retrained"] == list(range(min_slice, NUM_SLICES))
    assert result["resumed_from"] == f"slice{min_slice - 1}"


def test_rebuild_is_deterministic(tmp_path, monkeypatch):
    """Two rebuilds of the same subject from the same state produce identical
    weights. This is what makes Phase 4's spot-check able to detect a rebuild
    that did not do what its manifest claims."""
    digests = []
    for run in range(2):
        shard_dir, ckpt_dir = str(tmp_path / f"s{run}"), str(tmp_path / f"c{run}")
        monkeypatch.setattr(train_mod, "SHARD_DIR", shard_dir)
        monkeypatch.setattr(train_mod, "CHECKPOINT_DIR", ckpt_dir)
        monkeypatch.setattr(rebuild_mod, "SHARD_DIR", shard_dir)
        routing = train_mod.build(n_subjects=200, seed=7)
        for k in range(NUM_SHARDS):
            train_mod.train_shard(k)
        target = next(s for s in (f"C{i:07d}" for i in range(200)) if subject_ref(s) in routing)
        digests.append(rebuild_mod.rebuild(target)["result_weights"])
    assert digests[0] == digests[1]


def test_routing_entry_is_removed(built):
    routing, shard_dir = built
    target = next(s for s in (f"C{i:07d}" for i in range(200)) if subject_ref(s) in routing)

    rebuild_mod.rebuild(target)

    with open(os.path.join(shard_dir, "routing.json")) as f:
        assert subject_ref(target) not in json.load(f)


def test_unknown_subject_is_refused(built):
    with pytest.raises(KeyError):
        rebuild_mod.rebuild("C9999999")


def test_rebuild_emits_a_verifiable_certificate(built, monkeypatch):
    """Closes the loop: the manifest the engine actually produces must verify
    under the standalone verifier, not just a hand-built one from a fixture."""
    from nacl.signing import SigningKey
    from verify.verifier_cli import verify_certificate

    key = SigningKey.generate()
    monkeypatch.setenv("UNLEARNSHIELD_SIGNING_KEY", bytes(key).hex())

    routing, _ = built
    target = next(s for s in (f"C{i:07d}" for i in range(200)) if subject_ref(s) in routing)

    manifest = rebuild_mod.rebuild(target)["manifest"]

    ok, findings = verify_certificate(dict(manifest), key.verify_key)
    assert ok, findings
    assert manifest["subject_ref"] == subject_ref(target)


def test_emitted_certificate_names_a_root_the_subject_is_really_gone_from(built, monkeypatch):
    """Guards against a root computed before the purge -- which would verify
    cleanly while proving absence from a set the model never trained on."""
    from nacl.signing import SigningKey
    from verify.smt import build_root

    monkeypatch.setenv("UNLEARNSHIELD_SIGNING_KEY", bytes(SigningKey.generate()).hex())
    routing, _ = built
    target = next(s for s in (f"C{i:07d}" for i in range(200)) if subject_ref(s) in routing)
    shard = routing[subject_ref(target)]["shard"]

    manifest = rebuild_mod.rebuild(target)["manifest"]

    retained = set(train_mod.load_shard(shard)["subject_ref"].tolist())
    assert subject_ref(target) not in retained
    assert manifest["dataset_root"] == build_root(retained)
