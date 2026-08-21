"""The verifier's whole value is that it runs without the training system.

If verifying a certificate needs engine/, the database, or the operator's
environment, it stops being a proof and becomes the operator asserting their own
compliance and shipping a script that agrees with them.
"""

import ast
import json
import os
import shutil
import subprocess
import sys
from hashlib import sha256

from nacl.signing import SigningKey

from verify.manifest import build, canonical_bytes
from verify.smt import build_root, prove_absence

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FORBIDDEN = {"engine", "gateway", "worker", "config", "data", "scripts",
             "psycopg2", "torch", "sqlalchemy"}


def test_verify_package_imports_nothing_from_the_training_system():
    offenders = []
    for name in sorted(os.listdir(os.path.join(REPO, "verify"))):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(REPO, "verify", name)).read())
        for node in ast.walk(tree):
            roots = []
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = [node.module.split(".")[0]]
            offenders += [f"verify/{name} imports {r}" for r in roots if r in FORBIDDEN]
    assert offenders == []


def test_cli_verifies_a_manifest_in_a_directory_holding_only_verify(tmp_path):
    """Copy verify/ somewhere bare, hand it a manifest, and run it there."""
    shutil.copytree(os.path.join(REPO, "verify"), tmp_path / "verify")
    for junk in ("__pycache__",):
        shutil.rmtree(tmp_path / "verify" / junk, ignore_errors=True)

    population = [sha256(f"s{i}".encode()).hexdigest() for i in range(80)]
    target = sha256(b"erased").hexdigest()
    key = SigningKey.generate()
    manifest = build(
        subject_ref=target, shard=1, resumed_from="slice2",
        dataset_root=build_root(population),
        absence_proof=prove_absence(target, population),
        code_digest="sha256:img", config_digest="sha256:cfg",
        result_weights="sha256:w", model_version="v1",
        purged_at="2026-08-21T10:00:00Z", completed_at="2026-08-21T10:04:12Z",
    )
    manifest["signature"] = key.sign(canonical_bytes(manifest)).signature.hex()

    (tmp_path / "verify" / "public_key.hex").write_text(key.verify_key.encode().hex())
    (tmp_path / "cert.json").write_text(json.dumps(manifest))

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-m", "verify.verifier_cli", "cert.json"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VERIFIED" in result.stdout
    assert "Does not prove" in result.stdout


def test_cli_exits_nonzero_on_a_tampered_manifest(tmp_path):
    shutil.copytree(os.path.join(REPO, "verify"), tmp_path / "verify")
    shutil.rmtree(tmp_path / "verify" / "__pycache__", ignore_errors=True)

    population = [sha256(f"s{i}".encode()).hexdigest() for i in range(40)]
    target = sha256(b"erased").hexdigest()
    key = SigningKey.generate()
    manifest = build(
        subject_ref=target, shard=1, resumed_from="slice2",
        dataset_root=build_root(population),
        absence_proof=prove_absence(target, population),
        code_digest="d", config_digest="c", result_weights="w", model_version="v",
        purged_at="2026-08-21T10:00:00Z", completed_at="2026-08-21T10:04:12Z",
    )
    manifest["signature"] = key.sign(canonical_bytes(manifest)).signature.hex()
    manifest["shard"] = 4  # tamper after signing

    (tmp_path / "verify" / "public_key.hex").write_text(key.verify_key.encode().hex())
    (tmp_path / "cert.json").write_text(json.dumps(manifest))

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-m", "verify.verifier_cli", "cert.json"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "REJECTED" in result.stdout
