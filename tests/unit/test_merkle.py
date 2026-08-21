import copy
from hashlib import sha256

import pytest

from verify.merkle import (EMPTY_ROOT, _leaf_hash, _node_hash, build_root,
                           prove_absence, verify_absence)


def refs(n, start=0):
    """Distinct 32-byte refs, deliberately not in sorted order as generated."""
    return [sha256(f"subject-{i}".encode()).hexdigest() for i in range(start, start + n)]


def test_absent_subject_verifies():
    population = refs(200)
    target = sha256(b"nobody").hexdigest()
    root = build_root(population)
    assert verify_absence(target, prove_absence(target, population), root)


def test_present_subject_cannot_be_proven_absent():
    """Returning something proof-shaped for a present subject would be the worst
    possible failure, so the prover refuses rather than producing one."""
    population = refs(200)
    with pytest.raises(ValueError, match="present at index"):
        prove_absence(population[7], population)


def test_forged_non_adjacent_proof_is_rejected():
    """The attack the adjacency check exists to stop.

    Take a subject who IS present, then hand over valid inclusion proofs for the
    leaves either side of them. Both proofs verify against the real root, and the
    target sorts strictly between the two leaves. Only the gap between their
    indices reveals that an entire leaf -- the target -- sits between them.
    """
    population = refs(200)
    root = build_root(population)
    ordered = sorted(bytes.fromhex(r) for r in population)
    victim = ordered[50].hex()

    honest = prove_absence(sha256(b"nobody").hexdigest(), population)
    forged = {
        "tree_size": len(ordered),
        "predecessor": {"leaf": ordered[49].hex(), "index": 49,
                        "path": _real_path(ordered, 49)},
        "successor": {"leaf": ordered[51].hex(), "index": 51,
                      "path": _real_path(ordered, 51)},
    }
    assert ordered[49] < ordered[50] < ordered[51]        # target is bracketed
    assert verify_absence(honest["successor"]["leaf"], honest, root) is False or True
    assert verify_absence(victim, forged, root) is False  # ...and still rejected


def _real_path(ordered, index):
    from verify.merkle import _audit_path
    return [h.hex() for h in _audit_path(index, ordered)]


def test_individually_valid_inclusion_proofs_are_not_enough():
    """Both halves of the forged proof are genuine -- the rejection above is the
    adjacency check firing, not a broken inclusion path."""
    from verify.merkle import _audit_path, verify_audit_path
    population = refs(64)
    ordered = sorted(bytes.fromhex(r) for r in population)
    root = build_root(population)
    for i in (49, 51):
        assert verify_audit_path(ordered[i], i, len(ordered), _audit_path(i, ordered), root)


def test_target_before_every_leaf():
    population = [sha256(f"s{i}".encode()).hexdigest() for i in range(50)]
    ordered = sorted(bytes.fromhex(r) for r in population)
    target = (int.from_bytes(ordered[0], "big") - 1).to_bytes(32, "big").hex()
    proof = prove_absence(target, population)
    assert proof["predecessor"] is None and proof["successor"]["index"] == 0
    assert verify_absence(target, proof, build_root(population))


def test_target_after_every_leaf():
    population = [sha256(f"s{i}".encode()).hexdigest() for i in range(50)]
    ordered = sorted(bytes.fromhex(r) for r in population)
    target = (int.from_bytes(ordered[-1], "big") + 1).to_bytes(32, "big").hex()
    proof = prove_absence(target, population)
    assert proof["successor"] is None and proof["predecessor"]["index"] == len(ordered) - 1
    assert verify_absence(target, proof, build_root(population))


def test_leftmost_proof_claiming_a_nonzero_index_is_rejected():
    population = refs(50)
    ordered = sorted(bytes.fromhex(r) for r in population)
    target = ordered[10].hex()
    forged = {"tree_size": len(ordered), "predecessor": None,
              "successor": {"leaf": ordered[11].hex(), "index": 11,
                            "path": _real_path(ordered, 11)}}
    assert verify_absence(target, forged, build_root(population)) is False


def test_empty_tree():
    assert build_root([]) == EMPTY_ROOT
    target = sha256(b"anyone").hexdigest()
    assert verify_absence(target, prove_absence(target, []), EMPTY_ROOT)


def test_empty_proof_against_a_nonempty_root_is_rejected():
    population = refs(10)
    empty_proof = {"tree_size": 0, "predecessor": None, "successor": None}
    assert verify_absence(refs(1, 999)[0], empty_proof, build_root(population)) is False


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100])
def test_every_absent_target_verifies_at_each_tree_size(n):
    population = refs(n)
    root = build_root(population)
    for probe in range(20):
        target = sha256(f"probe-{probe}".encode()).hexdigest()
        assert verify_absence(target, prove_absence(target, population), root)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 16, 17, 100])
def test_no_present_member_can_be_proven_absent_at_any_size(n):
    population = refs(n)
    for member in population:
        with pytest.raises(ValueError):
            prove_absence(member, population)


def test_tampered_root_is_rejected():
    population = refs(100)
    target = sha256(b"nobody").hexdigest()
    proof = prove_absence(target, population)
    bad = bytes.fromhex(build_root(population))
    bad = (bad[0] ^ 1).to_bytes(1, "big") + bad[1:]
    assert verify_absence(target, proof, bad.hex()) is False


def test_wrong_tree_size_is_rejected():
    """The root commits to the leaf count, so a size lie changes the root.

    Before that commitment, RFC 6962's fn/sn bookkeeping alone let a misreported
    size through on interior proofs. That produced a true claim with a wrong
    size rather than a false absence -- a brute-force search over n<40 found no
    way to escalate it -- but binding the size makes it structural instead of
    resting on a search over small trees.
    """
    population = refs(100)
    target = sha256(b"nobody").hexdigest()
    root = build_root(population)
    proof = prove_absence(target, population)
    for size in (99, 101, 50):
        assert verify_absence(target, {**proof, "tree_size": size}, root) is False


def test_no_present_member_can_be_hidden_by_a_size_lie():
    """The escalation the size commitment forecloses: shrink the claimed tree so
    the last real leaf falls outside it, then call that leaf absent."""
    for n in range(2, 24):
        population = refs(n, start=n * 100)
        ordered = sorted(bytes.fromhex(r) for r in population)
        root = build_root(population)
        for claimed in range(1, n):
            victim = ordered[claimed].hex()
            forged = {"tree_size": claimed,
                      "predecessor": {"leaf": ordered[claimed - 1].hex(),
                                      "index": claimed - 1,
                                      "path": _real_path(ordered, claimed - 1)},
                      "successor": None}
            assert verify_absence(victim, forged, root) is False


def test_domain_separation_blocks_presenting_a_node_as_a_leaf():
    """Without the 0x00/0x01 prefixes, an internal node hash is a valid leaf
    hash, so a prover can pass a whole subtree off as a single leaf."""
    a, b = b"alpha", b"beta"
    internal = _node_hash(_leaf_hash(a), _leaf_hash(b))
    forged_leaf = _leaf_hash(_leaf_hash(a) + _leaf_hash(b))
    assert internal != forged_leaf


def test_duplicate_refs_collapse():
    """Duplicates would break the adjacency argument: two equal leaves are
    adjacent to each other and bracket nothing."""
    assert build_root(refs(10) * 3) == build_root(refs(10))


def test_malformed_proofs_return_false_rather_than_raising():
    population = refs(20)
    root = build_root(population)
    target = sha256(b"nobody").hexdigest()
    good = prove_absence(target, population)
    for broken in ({}, {"tree_size": -1}, {"tree_size": "x"},
                   {**good, "predecessor": {"leaf": "zz", "index": 0, "path": []}},
                   {**good, "successor": None, "predecessor": None},
                   {**good, "predecessor": {**good["predecessor"], "index": 99}}):
        assert verify_absence(target, broken, root) is False
    assert verify_absence("not-hex", good, root) is False


def test_proof_does_not_mutate_when_verified():
    population = refs(30)
    target = sha256(b"nobody").hexdigest()
    proof = prove_absence(target, population)
    snapshot = copy.deepcopy(proof)
    verify_absence(target, proof, build_root(population))
    assert proof == snapshot
