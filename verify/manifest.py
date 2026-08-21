"""Canonical manifest serialisation. No imports from engine/, gateway/, or the DB.

Canonical form is `sort_keys=True, separators=(",", ":"), ensure_ascii=True`.
Signature verification is meaningless if two semantically identical manifests
serialise to different bytes -- the verifier re-canonicalises the parsed object
and checks the signature against those bytes, so a manifest that round-trips
differently is a manifest that fails to verify.
"""

import json

REQUIRED_FIELDS = (
    "subject_ref", "shard", "resumed_from", "dataset_root", "absence_proof",
    "code_digest", "config_digest", "result_weights", "model_version",
    "purged_at", "completed_at",
)


def canonical_bytes(manifest: dict) -> bytes:
    missing = [f for f in REQUIRED_FIELDS if f not in manifest]
    if missing:
        raise ValueError(f"manifest missing required fields: {missing}")
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode()


def build(subject_ref: str, shard: int, resumed_from: str, dataset_root: str,
          absence_proof: dict, code_digest: str, config_digest: str,
          result_weights: str, model_version: str, purged_at: str,
          completed_at: str) -> dict:
    """Plain values in, plain dict out -- deliberately no engine types.

    This is the boundary that keeps verify/ independent of the training system.
    """
    return {
        "subject_ref": subject_ref,
        "shard": shard,
        "resumed_from": resumed_from,
        "dataset_root": dataset_root,
        "absence_proof": absence_proof,
        "code_digest": code_digest,
        "config_digest": config_digest,
        "result_weights": result_weights,
        "model_version": model_version,
        "purged_at": purged_at,
        "completed_at": completed_at,
    }
