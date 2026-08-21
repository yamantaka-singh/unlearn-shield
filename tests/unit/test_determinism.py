import os
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.determinism import DeterminismError, enforce_determinism, state_dict_digest
from scripts.spot_check_determinism import build_model, train_once

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_two_runs_are_byte_identical():
    assert state_dict_digest(train_once()) == state_dict_digest(train_once())


def test_thread_count_is_pinned():
    """The regression guard that seeding alone would not catch.

    Float addition is not associative, so a reduction split across 4 threads and
    the same reduction split across 8 produce different weights from identical
    inputs. Unpinning this is how determinism dies silently on a bigger host.
    """
    enforce_determinism(1)
    assert torch.get_num_threads() == 1
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_unseeded_training_diverges():
    """Negative control: without the harness the digests must not all agree.

    If this ever passes, the positive test above is proving nothing.
    """
    digests = set()
    for _ in range(5):
        torch.seed()  # re-randomise from OS entropy
        digests.add(state_dict_digest(build_model(12).state_dict()))
    assert len(digests) > 1


def test_missing_pythonhashseed_is_refused():
    with pytest.raises(DeterminismError, match="PYTHONHASHSEED"):
        env = dict(os.environ, PYTHONHASHSEED="random")
        raise_in_subprocess(env)


def raise_in_subprocess(env):
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "from config.determinism import enforce_determinism; enforce_determinism(1)" % REPO_ROOT],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise DeterminismError(result.stderr.strip().splitlines()[-1])


def test_spot_check_script_exits_zero():
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", "spot_check_determinism.py"), "--ci"],
        env=dict(os.environ, PYTHONHASHSEED="0"), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
