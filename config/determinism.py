"""Determinism harness. Import and call `enforce_determinism` before touching data.

The guarantee this project sells is that a retrained shard is a function of its
retained data and nothing else. Bit-identical weights are not that guarantee --
they are the cheap way to *audit* it, so a manifest can be spot-checked without
a second full training run.

Consequence: identity is only expected within a single `code_digest`. A torch
upgrade is allowed to change the weights; it is not allowed to change them for
a fixed code_digest. See docs/adr/0003-cpu-only-determinism.md.
"""

import hashlib
import os
import random

import numpy as np
import torch


class DeterminismError(RuntimeError):
    pass


def enforce_determinism(seed: int) -> torch.device:
    """Pin every source of run-to-run variation we know about. Returns the device.

    Raises DeterminismError if PYTHONHASHSEED was not set before interpreter start,
    which is the one source that cannot be fixed from inside the process.
    """
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise DeterminismError(
            "PYTHONHASHSEED must be 0 and can only be set before the interpreter "
            "starts. Re-run as: PYTHONHASHSEED=0 <command>"
        )

    # Thread count changes the split of every float reduction, and float addition
    # is not associative -- so 4 threads and 8 threads produce different weights
    # from identical inputs. This survives seeding, and it is why "deterministic
    # on my laptop" does not imply deterministic in a container scheduled onto a
    # host with a different core count.
    # ponytail: single-threaded; if shard training gets too slow, pin an explicit
    # thread count in config and treat it as part of code_digest, don't unpin it.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Only settable before the first parallel region; already-set is fine,
        # and a mismatch would have shown up in the spot-check.
        pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    return torch.device("cpu")  # GPU needs its own ADR, not a flag flip here


def state_dict_digest(state_dict) -> str:
    """sha256 over tensor bytes, key order fixed. The unit the spot-check compares."""
    h = hashlib.sha256()
    for key in sorted(state_dict):
        h.update(key.encode())
        h.update(state_dict[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()
