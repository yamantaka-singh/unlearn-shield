"""Train the same tiny shard twice and assert the weights are byte-identical.

Runs in CI on every push, and in production against a sampled fraction of real
rebuilds (Phase 4). The model here is a stand-in: when engine/model.py lands in
Phase 2, point `build_model` at it so the check exercises the real training path
rather than a proxy for it.
"""

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])

import numpy as np
import torch
from torch import nn

from config.determinism import enforce_determinism, state_dict_digest
from config.settings import SEED


def build_model(n_features: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_features, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 1),
    )


def train_once(seed: int = SEED, n_rows: int = 512, n_features: int = 12, epochs: int = 3) -> dict:
    enforce_determinism(seed)
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(n_rows, n_features)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, 2, size=(n_rows, 1)), dtype=torch.float32)

    model = build_model(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=64,
        shuffle=True,
        num_workers=0,  # workers reseed independently; adding them needs worker_init_fn
        generator=torch.Generator().manual_seed(seed),
    )

    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model.state_dict()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="terse output, exit code is the result")
    args = parser.parse_args()

    a, b = state_dict_digest(train_once()), state_dict_digest(train_once())
    if a == b:
        print(f"PASS deterministic: {a[:16]}")
        return 0
    print(f"FAIL non-deterministic:\n  run 1 {a}\n  run 2 {b}")
    if not args.ci:
        print("\nUsual causes: thread count not pinned, PYTHONHASHSEED unset,")
        print("a DataLoader worker without worker_init_fn, or a floated dependency.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
