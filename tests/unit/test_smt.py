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


def test_tree_matches_the_single_shot_functions_and_verifies():
    """SparseMerkleTree exists purely as a performance fix (one build serving
    the root and every proof, instead of a rebuild per call). A performance fix
    that changes a root is a correctness disaster: every certificate ever
    issued verifies against a recorded dataset_root, so the two paths must
    agree byte for byte, not merely 'both verify'."""
    import secrets

    from verify.smt import (SparseMerkleTree, build_root, prove_absence,
                            verify_absence)

    for n in (0, 1, 2, 3, 17, 400):
        refs = [secrets.token_hex(32) for _ in range(n)]
        tree = SparseMerkleTree(refs)
        assert tree.root == build_root(refs), f"root diverged at n={n}"
        for _ in range(5):
            target = secrets.token_hex(32)
            assert tree.prove_absence(target) == prove_absence(target, refs), \
                f"proof diverged at n={n}"
            assert verify_absence(target, tree.prove_absence(target), tree.root)


def test_tree_refuses_to_prove_absence_of_a_present_key():
    import secrets

    import pytest

    from verify.smt import SparseMerkleTree

    refs = [secrets.token_hex(32) for _ in range(50)]
    tree = SparseMerkleTree(refs)
    with pytest.raises(ValueError):
        tree.prove_absence(refs[0])


def test_a_proof_does_not_verify_against_another_trees_root():
    """Negative control. Without it the pair above would pass for a tree that
    returned the same constant for everything."""
    import secrets

    from verify.smt import SparseMerkleTree, verify_absence

    a = SparseMerkleTree([secrets.token_hex(32) for _ in range(50)])
    b = SparseMerkleTree([secrets.token_hex(32) for _ in range(50)])
    target = secrets.token_hex(32)
    assert verify_absence(target, a.prove_absence(target), a.root)
    assert not verify_absence(target, a.prove_absence(target), b.root)


def test_remove_matches_a_fresh_build_over_the_remaining_keys():
    """The whole safety requirement for incremental removal: it must produce
    the identical root and proofs a from-scratch build over the survivors
    would, not merely 'a plausible-looking' one -- a manifest signs whatever
    root this returns."""
    import secrets

    from verify.smt import SparseMerkleTree, build_root, prove_absence

    for n, remove_n in ((1, 1), (2, 1), (5, 2), (200, 1), (200, 50), (200, 199)):
        population = [secrets.token_hex(32) for _ in range(n)]
        tree = SparseMerkleTree(population)
        removed = population[:remove_n]
        survivors = population[remove_n:]

        after = tree.remove(removed)
        assert after.root == build_root(survivors), f"n={n} remove_n={remove_n}"

        for _ in range(5):
            probe = secrets.token_hex(32)
            assert after.prove_absence(probe) == prove_absence(probe, survivors)
        for gone in removed:
            assert after.prove_absence(gone) == prove_absence(gone, survivors)


def test_repeated_removal_rounds_match_a_fresh_build_each_time():
    """Simulates the real usage pattern: many separate erasure rounds against
    the same shard, each removing a few keys from whatever remains. Catches
    any bug in the recursive removal walker that a single-round test, however
    many sizes it tries, would not -- e.g. incorrect handling of a subtree
    that collapses to a singleton and is then removed again later."""
    import secrets

    from verify.smt import SparseMerkleTree, build_root

    population = [secrets.token_hex(32) for _ in range(300)]
    tree = SparseMerkleTree(population)
    remaining = list(population)
    rng = secrets.SystemRandom()

    for _ in range(15):
        if len(remaining) <= 1:
            break
        batch = rng.sample(remaining, k=min(5, len(remaining) - 1))
        tree = tree.remove(batch)
        remaining = [k for k in remaining if k not in set(batch)]
        assert tree.root == build_root(remaining)


def test_remove_raises_on_a_key_that_is_not_present():
    import secrets

    from verify.smt import SparseMerkleTree

    population = [secrets.token_hex(32) for _ in range(50)]
    tree = SparseMerkleTree(population)
    stranger = secrets.token_hex(32)
    with pytest.raises(ValueError):
        tree.remove([stranger])


def test_remove_of_all_but_present_keys_raises_before_mutating_anything():
    """One bad ref in a batch must not partially apply -- a caller retries the
    whole batch on failure, and a tree that already dropped some of the valid
    refs would silently double-process them next time (or, worse, keep
    serving a tree that matches neither the old nor the new state)."""
    import secrets

    from verify.smt import SparseMerkleTree

    population = [secrets.token_hex(32) for _ in range(10)]
    tree = SparseMerkleTree(population)
    stranger = secrets.token_hex(32)
    before_root = tree.root

    with pytest.raises(ValueError):
        tree.remove([population[0], stranger])
    assert tree.root == before_root, "a failed remove() must not mutate the tree in place"


def test_fingerprint_is_order_independent_and_set_sensitive():
    import secrets

    from verify.smt import fingerprint_of

    population = [secrets.token_hex(32) for _ in range(50)]
    shuffled = list(reversed(population))
    assert fingerprint_of(population) == fingerprint_of(shuffled)

    changed = population[:-1] + [secrets.token_hex(32)]
    assert fingerprint_of(population) != fingerprint_of(changed)


def test_tree_for_shard_reuses_a_cache_hit_without_rebuilding():
    """The mechanism, not just the math: proves the expensive full-build path
    is actually SKIPPED on a cache hit, by counting calls to the function that
    does the expensive work -- a test that only checked the resulting root
    could pass even if caching silently never engaged (a fresh build gives
    the same root as a reused one, by construction)."""
    import secrets

    import verify.smt as smt_mod
    from verify.smt import cache_tree, clear_tree_cache, tree_for_shard

    clear_tree_cache()
    population = [secrets.token_hex(32) for _ in range(80)]
    calls = {"n": 0}
    real_build = smt_mod._build

    def counting_build(keys, depth):
        calls["n"] += 1
        return real_build(keys, depth)

    smt_mod._build = counting_build
    try:
        tree = tree_for_shard(999001, population)
        # _build recurses, so a real build calls it many times (once per node
        # visited), not once -- the assertion that matters is that a MISS does
        # real work at all, and a HIT (below) does none.
        assert calls["n"] > 0, "first call for a shard must do a real build"
        cache_tree(999001, tree)

        calls["n"] = 0
        again = tree_for_shard(999001, population)
        assert calls["n"] == 0, "a fingerprint-matching cache hit must not call _build at all"
        assert again is tree
    finally:
        smt_mod._build = real_build
        clear_tree_cache()


def test_tree_for_shard_falls_back_to_a_fresh_build_on_a_mismatch():
    """The safety net: a cache entry that no longer matches reality (a
    different worker rebuilt this shard, a restart lost in-memory state,
    whatever the cause) must be silently DISCARDED, not trusted. Falling back
    to a full build is the only acceptable behaviour here -- reusing a stale
    tree would let a certificate commit to a root that does not describe what
    is actually on disk."""
    import secrets

    from verify.smt import SparseMerkleTree, cache_tree, clear_tree_cache, tree_for_shard

    clear_tree_cache()
    stale_population = [secrets.token_hex(32) for _ in range(30)]
    cache_tree(999002, SparseMerkleTree(stale_population))

    actual_population = [secrets.token_hex(32) for _ in range(30)]  # unrelated keys
    tree = tree_for_shard(999002, actual_population)
    assert tree.fingerprint == SparseMerkleTree(actual_population).fingerprint
    clear_tree_cache()


def test_tree_for_shard_with_empty_cache_builds_fresh():
    import secrets

    from verify.smt import clear_tree_cache, tree_for_shard

    clear_tree_cache()
    population = [secrets.token_hex(32) for _ in range(10)]
    tree = tree_for_shard(999003, population)
    assert tree.fingerprint is not None
    clear_tree_cache()
