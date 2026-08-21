import json
from hashlib import sha256

import pytest
from nacl.signing import SigningKey

from verify.manifest import REQUIRED_FIELDS, build, canonical_bytes
# Deliberately the OLD sorted-Merkle scheme. engine/rebuild.py now issues
# sparse-tree proofs, so these cases double as the regression test that
# certificates issued before that switch still verify -- a certificate outlives
# the code that made it. test_smt_scheme_also_verifies covers the current one.
from verify.merkle import build_root, prove_absence
from verify.sign import verify_manifest
from verify.verifier_cli import verify_certificate


@pytest.fixture
def signed():
    population = [sha256(f"s{i}".encode()).hexdigest() for i in range(64)]
    target = sha256(b"erased").hexdigest()
    key = SigningKey.generate()
    m = build(
        subject_ref=target, shard=2, resumed_from="slice3",
        dataset_root=build_root(population), absence_proof=prove_absence(target, population),
        code_digest="sha256:img", config_digest="sha256:cfg",
        result_weights="sha256:w", model_version="v47",
        purged_at="2026-08-21T10:00:00Z", completed_at="2026-08-21T10:04:12Z",
    )
    m["signature"] = key.sign(canonical_bytes(m)).signature.hex()
    return m, key.verify_key


def test_valid_certificate_verifies(signed):
    m, pub = signed
    ok, findings = verify_certificate(dict(m), pub)
    assert ok, findings


def test_key_order_does_not_change_the_signed_bytes():
    """Two semantically identical manifests must serialise identically, or a
    signature means nothing."""
    fields = {f: "x" for f in REQUIRED_FIELDS}
    a = dict(sorted(fields.items()))
    b = dict(reversed(list(fields.items())))
    assert canonical_bytes(a) == canonical_bytes(b)


@pytest.mark.parametrize("field", ["subject_ref", "shard", "dataset_root", "code_digest",
                                   "result_weights", "model_version", "resumed_from",
                                   "config_digest", "purged_at", "completed_at"])
def test_tampering_with_any_field_is_detected(signed, field):
    m, pub = signed
    m = dict(m)
    m[field] = "tampered" if isinstance(m[field], str) else 999
    ok, findings = verify_certificate(m, pub)
    assert not ok and "SIGNATURE INVALID" in findings[0]


def test_tampering_with_the_absence_proof_is_detected(signed):
    m, pub = signed
    m = dict(m)
    m["absence_proof"] = {**m["absence_proof"], "tree_size": 1}
    ok, _ = verify_certificate(m, pub)
    assert not ok


def test_signature_from_another_key_is_rejected(signed):
    m, _ = signed
    ok, findings = verify_certificate(dict(m), SigningKey.generate().verify_key)
    assert not ok and "SIGNATURE INVALID" in findings[0]


def test_unsigned_manifest_is_rejected(signed):
    m, pub = signed
    m = {k: v for k, v in m.items() if k != "signature"}
    ok, findings = verify_certificate(m, pub)
    assert not ok and "no signature" in findings[0]


def test_missing_field_is_rejected(signed):
    m, pub = signed
    m = {k: v for k, v in m.items() if k != "dataset_root"}
    ok, findings = verify_certificate(m, pub)
    assert not ok and "missing required fields" in findings[0]


def test_a_correctly_signed_but_false_absence_claim_is_rejected(signed):
    """The case that matters: the operator holds the signing key, so a signature
    only proves they authored the claim. The absence proof is what makes it
    checkable -- signing a lie must still fail."""
    population = [sha256(f"s{i}".encode()).hexdigest() for i in range(64)]
    present = population[10]
    key = SigningKey.generate()
    target = sha256(b"decoy").hexdigest()
    m = build(
        subject_ref=present,  # claim a subject who is still in the retained set
        shard=2, resumed_from="slice3", dataset_root=build_root(population),
        absence_proof=prove_absence(target, population),  # proof about someone else
        code_digest="d", config_digest="c", result_weights="w", model_version="v",
        purged_at="2026-08-21T10:00:00Z", completed_at="2026-08-21T10:04:12Z",
    )
    m["signature"] = key.sign(canonical_bytes(m)).signature.hex()

    ok, findings = verify_certificate(m, key.verify_key)
    assert not ok
    assert any("ABSENCE PROOF INVALID" in f for f in findings)


def test_canonical_bytes_refuses_incomplete_manifests():
    with pytest.raises(ValueError, match="missing required fields"):
        canonical_bytes({"shard": 1})


def test_verify_manifest_helper_matches_the_cli(signed):
    m, pub = signed
    sig = m.pop("signature")
    assert verify_manifest(m, sig, pub)
    assert not verify_manifest({**m, "shard": 99}, sig, pub)


def test_smt_scheme_also_verifies():
    """The scheme engine/rebuild.py actually issues today."""
    from verify import smt

    population = [sha256(f"s{i}".encode()).hexdigest() for i in range(64)]
    target = sha256(b"erased").hexdigest()
    key = SigningKey.generate()
    m = build(
        subject_ref=target, shard=2, resumed_from="slice3",
        dataset_root=smt.build_root(population),
        absence_proof=smt.prove_absence(target, population),
        code_digest="sha256:img", config_digest="sha256:cfg",
        result_weights="sha256:w", model_version="v47",
        purged_at="2026-08-21T10:00:00Z", completed_at="2026-08-21T10:04:12Z",
    )
    m["signature"] = key.sign(canonical_bytes(m)).signature.hex()

    ok, findings = verify_certificate(dict(m), key.verify_key)
    assert ok, findings
    assert any("smt-256" in f for f in findings)


def test_smt_certificate_with_a_false_claim_is_rejected():
    """Signed correctly, but claiming absence of a subject still in the set."""
    from verify import smt

    population = [sha256(f"s{i}".encode()).hexdigest() for i in range(64)]
    key = SigningKey.generate()
    decoy = sha256(b"decoy").hexdigest()
    m = build(
        subject_ref=population[10], shard=2, resumed_from="slice3",
        dataset_root=smt.build_root(population),
        absence_proof=smt.prove_absence(decoy, population),
        code_digest="d", config_digest="c", result_weights="w", model_version="v",
        purged_at="2026-08-21T10:00:00Z", completed_at="2026-08-21T10:04:12Z",
    )
    m["signature"] = key.sign(canonical_bytes(m)).signature.hex()

    ok, findings = verify_certificate(m, key.verify_key)
    assert not ok
    assert any("ABSENCE PROOF INVALID" in f for f in findings)
