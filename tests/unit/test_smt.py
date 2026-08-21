import copy
from hashlib import sha256

import pytest

from verify import merkle
from verify.smt import DEPTH, EMPTY_ROOT, build_root, prove_absence, verify_absence


def refs(n, start=0):
    return [sha256(f"subject-{i}".encode()).hexdigest() for i in range(start, start + n)]


def test_absent_subject_verifies():
    population = refs(200)
    target = sha256(b"nobody").hexdigest()
    assert verify_absence(target, prove_absence(target, population), build_root(population))


def test_present_subject_cannot_be_proven_absent():
    population = refs(200)
    with pytest.raises(ValueError, match="present"):
        prove_absence(population[7], population)


def test_proof_names_no_other_subject():
    """The reason this module exists.

    A sorted-Merkle absence proof carries the target's two sort-order
    neighbours in cleartext, so every certificate discloses two real
    subject_refs. Here the proof is sibling subtree hashes: a sibling covering
    populated ground commits to those subjects without naming them.
    """
    population = refs(300)
    target = sha256(b"nobody").hexdigest()

    smt_blob = repr(prove_absence(target, population))
    assert not any(ref in smt_blob for ref in population)

    sorted_blob = repr(merkle.prove_absence(target, population))
    leaked = [ref for ref in population if ref in sorted_blob]
    assert len(leaked) == 2, "the old scheme leaks exactly the two neighbours"


def test_proof_is_compressed_not_256_siblings():
    """Default (all-empty) siblings are recomputable by the verifier, so only
    the informative ones travel. Sending all 256 would work and be ~8KB of
    noise per certificate."""
    population = refs(400)
    target = sha256(b"nobody").hexdigest()
    proof = prove_absence(target, population)
    assert len(proof["siblings"]) < 32
    assert verify_absence(target, proof, build_root(population))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 8, 100])
def test_absence_holds_at_several_population_sizes(n):
    population = refs(n)
    root = build_root(population)
    for probe in range(10):
        target = sha256(f"probe-{probe}".encode()).hexdigest()
        assert verify_absence(target, prove_absence(target, population), root)


@pytest.mark.parametrize("n", [1, 2, 3, 8, 60])
def test_no_member_can_be_proven_absent(n):
    population = refs(n, start=n * 1000)
    for member in population:
        with pytest.raises(ValueError):
            prove_absence(member, population)


def test_empty_tree():
    target = sha256(b"anyone").hexdigest()
    assert build_root([]) == EMPTY_ROOT
    assert verify_absence(target, prove_absence(target, []), EMPTY_ROOT)


def test_tampered_root_is_rejected():
    population = refs(100)
    target = sha256(b"nobody").hexdigest()
    proof = prove_absence(target, population)
    bad = bytearray(bytes.fromhex(build_root(population)))
    bad[0] ^= 1
    assert verify_absence(target, proof, bytes(bad).hex()) is False


def test_proof_for_one_target_does_not_verify_for_another():
    """Siblings are bound to a path, and the path is the subject_ref itself."""
    population = refs(100)
    a = sha256(b"absent-a").hexdigest()
    b = sha256(b"absent-b").hexdigest()
    root = build_root(population)
    assert verify_absence(a, prove_absence(a, population), root)
    assert verify_absence(b, prove_absence(a, population), root) is False


def test_forged_sibling_is_rejected():
    population = refs(100)
    target = sha256(b"nobody").hexdigest()
    root = build_root(population)
    proof = prove_absence(target, population)
    depth = next(iter(proof["siblings"]))
    forged = copy.deepcopy(proof)
    forged["siblings"][depth] = sha256(b"made up").hexdigest()
    assert verify_absence(target, forged, root) is False


def test_dropping_a_sibling_is_rejected():
    """A dropped sibling silently becomes the empty default, which would let a
    prover erase a populated subtree from the computation."""
    population = refs(100)
    target = sha256(b"nobody").hexdigest()
    root = build_root(population)
    proof = prove_absence(target, population)
    stripped = copy.deepcopy(proof)
    stripped["siblings"].pop(next(iter(stripped["siblings"])))
    assert verify_absence(target, stripped, root) is False


def test_malformed_proofs_return_false_rather_than_raising():
    population = refs(50)
    root = build_root(population)
    target = sha256(b"nobody").hexdigest()
    good = prove_absence(target, population)
    for broken in ({}, {"scheme": "smt-256"}, {"scheme": "wrong", "siblings": {}},
                   {"scheme": "smt-256", "siblings": []},
                   {"scheme": "smt-256", "siblings": {"0": "zz"}},
                   {"scheme": "smt-256", "siblings": {"999": "ab" * 32}},
                   {"scheme": "smt-256", "siblings": {"0": "ab"}}):
        assert verify_absence(target, broken, root) is False
    assert verify_absence("not-hex", good, root) is False
    assert verify_absence("ab" * 16, good, root) is False  # 128-bit, wrong width


def test_verification_does_not_mutate_the_proof():
    population = refs(30)
    target = sha256(b"nobody").hexdigest()
    proof = prove_absence(target, population)
    snapshot = copy.deepcopy(proof)
    verify_absence(target, proof, build_root(population))
    assert proof == snapshot


def test_depth_matches_subject_ref_width():
    """subject_ref is HMAC-SHA256, so the ref itself is the path. If one changes
    the other must."""
    from config.settings import subject_ref
    assert len(bytes.fromhex(subject_ref("C0000001"))) * 8 == DEPTH
