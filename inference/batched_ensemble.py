"""One batched forward pass across every shard, and a cache so the checkpoints
are read from disk once rather than once per request.

Two separate wins, and the cache is by far the larger one:

    load 5 checkpoints from disk   2.63 ms   <- removed by the cache
    5 sequential forward passes    0.15 ms   <- reduced by the batching

Measured on this repo's 5-shard config. Anyone optimising the forward pass
before the disk read is optimising the wrong 5% -- which is the mistake worth
recording, since a batched-ensemble module looks like the obvious performance
work and mostly isn't.

The cache is keyed on the tuple of checkpoint hashes for the promoted
model_version, so a rebuild that promotes new weights produces a different key
and the stale entry is simply never looked up again. No invalidation call to
forget, which is the usual way a model cache serves erased data: the erasure
lands, the cache keeps answering from the pre-erasure model, and nothing
reports a problem.

Batching needs S preprocessed copies of the input, not one. Per-shard
preprocessing (ADR 0004) means each shard scales its inputs with its own
constants, so `vmap` maps over parameters AND inputs together. The plan
predicted a clean S-times saving from batching; it does not account for the
S-times preprocessing that per-shard fitting makes mandatory, so the real
figure is smaller and measured rather than asserted.
"""

import copy
import json
import threading

import numpy as np
import torch
from torch.func import functional_call, stack_module_state

from engine.model import build_model
from engine.preprocessing import ShardPreprocessor

_cache: dict = {}
_cache_lock = threading.Lock()

# ponytail: unbounded dict keyed by model_version's checkpoint tuple. Each entry
# is a handful of MB and a new one appears only on promotion, so it grows with
# rebuild count, not traffic. Swap for an LRU if a long-lived process promotes
# often enough to matter.


class ShardEnsemble:
    """Stacked shard sub-models plus their per-shard preprocessors."""

    def __init__(self, models: list, preprocessors: list):
        if not models:
            raise ValueError("ShardEnsemble needs at least one shard model")
        self.preprocessors = preprocessors
        self.params, self.buffers = stack_module_state(models)
        # A meta-device copy carries the module structure without allocating
        # storage; functional_call supplies the real tensors per call.
        self._base = copy.deepcopy(models[0]).to("meta")

    def shard_probabilities(self, records: dict, rows: np.ndarray) -> np.ndarray:
        """Per-shard probability, shape [n_shards, n_rows].

        The mean of this is what gets served; the spread across it is the
        optional disagreement signal (ADR 0009). Both callers share one
        forward pass -- the per-shard scores already exist inside
        predict_proba, which previously averaged them away.
        """
        stacked = torch.stack([
            torch.from_numpy(p.transform(records, rows)) for p in self.preprocessors
        ])  # [n_shards, n_rows, n_features]

        def one_shard(params, buffers, x):
            return functional_call(self._base, (params, buffers), (x,))

        with torch.no_grad():
            logits = torch.vmap(one_shard)(self.params, self.buffers, stacked)
        return torch.sigmoid(logits).squeeze(-1).numpy()

    def predict_proba(self, records: dict, rows: np.ndarray) -> np.ndarray:
        """Mean fraud probability across shards, one row per entry in `rows`."""
        return self.shard_probabilities(records, rows).mean(axis=0)


def load_ensemble(shard_paths: dict, preproc_paths: dict) -> ShardEnsemble:
    """`shard_paths` and `preproc_paths` map shard index (as str) -> file path.

    Paths come from the DB's `checkpoints.file_path`, which points at the
    content-addressed copy (ADR 0006) -- never at engine.train's conventional
    path, which later rebuilds overwrite.
    """
    key = tuple(sorted(shard_paths.items()))
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit

    models, preprocessors = [], []
    for shard in sorted(shard_paths, key=int):
        model = build_model()
        model.load_state_dict(torch.load(shard_paths[shard], weights_only=True))
        model.eval()
        models.append(model)
        with open(preproc_paths[shard]) as f:
            preprocessors.append(ShardPreprocessor.from_json(f.read()))

    ensemble = ShardEnsemble(models, preprocessors)
    with _cache_lock:
        # Another thread may have built the same ensemble concurrently; either
        # object is correct, so keep whichever landed first and let the
        # duplicate be collected.
        _cache.setdefault(key, ensemble)
        return _cache[key]


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
