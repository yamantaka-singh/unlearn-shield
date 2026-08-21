"""Fixed MLP, identical across every shard.

Identical is a requirement, not a convenience: Phase 5 stacks these submodels
into one batched forward pass, which needs matching parameter shapes. Any change
here is a change to every shard's architecture at once, so it invalidates every
existing checkpoint.
"""

from torch import nn

from engine.preprocessing import N_FEATURES


def build_model(n_features: int = N_FEATURES) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_features, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )
