import numpy as np
import pytest
import torch

from data.synth import generate
from engine.model import build_model
from engine.preprocessing import ShardPreprocessor
from inference.batched_ensemble import ShardEnsemble, clear_cache, load_ensemble


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_cache()
    yield
    clear_cache()


def _shard_parts(n_shards=5, seed=0):
    models, preprocessors, records = [], [], []
    for s in range(n_shards):
        torch.manual_seed(seed + s)
        model = build_model()
        model.eval()
        models.append(model)
        r = generate(n_subjects=40, seed=seed + s)
        rows = np.arange(len(r["step"]))
        preprocessors.append(ShardPreprocessor.fit(r, rows))
        records.append(r)
    return models, preprocessors, records[0]


def _sequential(models, preprocessors, records, rows):
    """What gateway/routes/predict.py did before batching: one forward pass per
    shard, each on its own preprocessed copy, averaged."""
    probs = []
    for model, preproc in zip(models, preprocessors):
        x = torch.from_numpy(preproc.transform(records, rows))
        with torch.no_grad():
            probs.append(torch.sigmoid(model(x)).squeeze(-1).numpy())
    return np.mean(probs, axis=0)


def test_batched_matches_sequential():
    """The whole point. A faster ensemble that returns a different number is
    not an optimisation, it is a second model nobody agreed to deploy."""
    models, preprocessors, records = _shard_parts()
    rows = np.arange(16)
    expected = _sequential(models, preprocessors, records, rows)
    actual = ShardEnsemble(models, preprocessors).predict_proba(records, rows)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("n_rows", [1, 2, 64])
def test_batched_matches_sequential_at_several_batch_sizes(n_rows):
    models, preprocessors, records = _shard_parts(seed=7)
    rows = np.arange(n_rows)
    expected = _sequential(models, preprocessors, records, rows)
    actual = ShardEnsemble(models, preprocessors).predict_proba(records, rows)
    assert actual.shape == (n_rows,)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_each_shard_uses_its_own_preprocessor():
    """Per-shard preprocessing (ADR 0004) is the isolation guarantee. If
    batching collapsed to one shared input tensor it would be faster, wrong,
    and silently so -- the shape is identical either way."""
    models, preprocessors, records = _shard_parts(seed=3)
    rows = np.arange(8)
    with_real = ShardEnsemble(models, preprocessors).predict_proba(records, rows)
    all_same = ShardEnsemble(models, [preprocessors[0]] * len(models)).predict_proba(records, rows)
    assert not np.allclose(with_real, all_same)


def test_probabilities_are_in_range():
    models, preprocessors, records = _shard_parts(seed=5)
    out = ShardEnsemble(models, preprocessors).predict_proba(records, np.arange(32))
    assert np.isfinite(out).all()
    assert (out >= 0).all() and (out <= 1).all()


def test_empty_model_list_is_refused():
    with pytest.raises(ValueError, match="at least one shard"):
        ShardEnsemble([], [])


def test_cache_returns_the_same_object_for_the_same_checkpoints(tmp_path):
    models, preprocessors, _ = _shard_parts(seed=9)
    shard_paths, preproc_paths = {}, {}
    for i, (model, preproc) in enumerate(zip(models, preprocessors)):
        mp = tmp_path / f"m{i}.pt"
        torch.save(model.state_dict(), mp)
        pp = tmp_path / f"p{i}.json"
        pp.write_text(preproc.to_json())
        shard_paths[str(i)] = str(mp)
        preproc_paths[str(i)] = str(pp)

    first = load_ensemble(shard_paths, preproc_paths)
    assert load_ensemble(shard_paths, preproc_paths) is first


def test_different_checkpoints_are_a_different_cache_entry(tmp_path):
    """A promotion changes the checkpoint hashes, so it changes the key and the
    stale entry simply stops being reachable. Without that, a cache would keep
    serving a model an erasure was supposed to remove data from, while every
    job row says done."""
    models, preprocessors, _ = _shard_parts(seed=11)
    paths = []
    for tag in ("a", "b"):
        shard_paths, preproc_paths = {}, {}
        for i, (model, preproc) in enumerate(zip(models, preprocessors)):
            mp = tmp_path / f"{tag}{i}.pt"
            torch.save(model.state_dict(), mp)
            pp = tmp_path / f"{tag}{i}.json"
            pp.write_text(preproc.to_json())
            shard_paths[str(i)] = str(mp)
            preproc_paths[str(i)] = str(pp)
        paths.append((shard_paths, preproc_paths))

    assert load_ensemble(*paths[0]) is not load_ensemble(*paths[1])


def test_predict_proba_is_exactly_the_mean_of_shard_probabilities():
    """ADR 0009 refactored predict_proba to delegate to shard_probabilities so
    the optional disagreement check could reuse the same forward pass. That
    refactor must not have changed a single served score -- this is the
    regression guard on "purely additive"."""
    models, preprocessors, records = _shard_parts(seed=21)
    rows = np.arange(24)
    ensemble = ShardEnsemble(models, preprocessors)

    per_shard = ensemble.shard_probabilities(records, rows)
    assert per_shard.shape == (len(models), len(rows))
    np.testing.assert_array_equal(ensemble.predict_proba(records, rows),
                                  per_shard.mean(axis=0))


def test_shard_probabilities_are_per_shard_not_broadcast():
    """If this returned the same row repeated, spread would be identically
    zero and the disagreement feature would silently never fire."""
    models, preprocessors, records = _shard_parts(seed=23)
    per_shard = ShardEnsemble(models, preprocessors).shard_probabilities(records, np.arange(8))
    assert per_shard.std(axis=0).sum() > 0
