"""Sorted Merkle tree and non-inclusion proofs. RFC 6962 hashing, no dependencies.

Three choices here are load-bearing. Each has a well-known failure mode that
produces a proof system which looks correct and is not.

**Domain separation.** Leaves hash as SHA256(0x00 || data), internal nodes as
SHA256(0x01 || left || right). Without the prefixes an internal node's hash is a
valid leaf hash, so a prover can present a subtree as a leaf and prove membership
of data that was never in the tree.

**RFC 6962 splitting, not odd-node duplication.** The common implementation
duplicates the last node when a level has an odd count, which lets two different
leaf sets produce the same root (the Bitcoin CVE-2012-2459 shape). RFC 6962
splits at the largest power of two below n instead, which is unambiguous.

**Tree size is bound into every proof.** The audit path verifier takes (index,
tree_size) and reconstructs the root from them. A proof that did not bind
tree_size could be replayed against a different tree where the same index means
a different position.

Leaves are distinct `subject_ref` values, not record ids. The deletion unit here
is a subject, so subject-level leaves make "this subject is absent" exactly one
non-inclusion proof. Record-level leaves would require proving every one of a
subject's records absent without knowing how many they had.
"""

from hashlib import sha256

def _commit(tree_size: int, mth_root: bytes) -> str:
    """Bind the leaf count into the published root.

    RFC 6962's audit-path check tracks tree size through its fn/sn bookkeeping,
    and a brute-force search over n<40 found no way to turn a size lie into a
    false absence proof. But that is evidence over a bounded range, not a
    property. Committing to the size makes any misreported tree_size change the
    root outright, so the guarantee holds by construction rather than by an
    argument that would need re-checking after every edit here.
    """
    return sha256(b"\x02" + tree_size.to_bytes(8, "big") + mth_root).hexdigest()


EMPTY_ROOT = _commit(0, sha256(b"").digest())


def _leaf_hash(data: bytes) -> bytes:
    return sha256(b"\x00" + data).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return sha256(b"\x01" + left + right).digest()


def _split(n: int) -> int:
    """Largest power of two strictly less than n."""
    return 1 << ((n - 1).bit_length() - 1)


def _mth(leaves: list[bytes]) -> bytes:
    # ponytail: recomputes subtrees per call, O(n log n) hashes per proof.
    # Fine at shard scale (~1e3 subjects); memoise levels if a shard reaches 1e6.
    n = len(leaves)
    if n == 0:
        return sha256(b"").digest()
    if n == 1:
        return _leaf_hash(leaves[0])
    k = _split(n)
    return _node_hash(_mth(leaves[:k]), _mth(leaves[k:]))


def _audit_path(index: int, leaves: list[bytes]) -> list[bytes]:
    n = len(leaves)
    if n <= 1:
        return []
    k = _split(n)
    if index < k:
        return _audit_path(index, leaves[:k]) + [_mth(leaves[k:])]
    return _audit_path(index - k, leaves[k:]) + [_mth(leaves[:k])]


def _sorted_leaves(subject_refs) -> list[bytes]:
    """Deduplicated and sorted. Duplicates would break the adjacency argument:
    two equal leaves are adjacent to each other, leaving a gap that brackets
    nothing."""
    return sorted({bytes.fromhex(r) for r in subject_refs})


def build_root(subject_refs) -> str:
    leaves = _sorted_leaves(subject_refs)
    return _commit(len(leaves), _mth(leaves))


def reconstruct_root(leaf: bytes, index: int, tree_size: int,
                     path: list[bytes]) -> bytes | None:
    """RFC 6962 section 2.1.1. Rebuilds the tree hash from (leaf, index, size)."""
    if index >= tree_size or index < 0:
        return None
    node = _leaf_hash(leaf)
    fn, sn = index, tree_size - 1
    for sibling in path:
        if sn == 0:
            return None
        if (fn & 1) or (fn == sn):
            node = _node_hash(sibling, node)
            while fn != 0 and not (fn & 1):
                fn >>= 1
                sn >>= 1
        else:
            node = _node_hash(node, sibling)
        fn >>= 1
        sn >>= 1
    return node if sn == 0 else None


def verify_audit_path(leaf: bytes, index: int, tree_size: int,
                      path: list[bytes], committed_root: str) -> bool:
    node = reconstruct_root(leaf, index, tree_size, path)
    return node is not None and _commit(tree_size, node) == committed_root


def prove_absence(target_ref: str, subject_refs) -> dict:
    """Prove `target_ref` is not among `subject_refs`.

    Raises ValueError if it IS present -- a caller asking for an absence proof
    of a present subject has a bug upstream, and returning something
    proof-shaped would be the worst possible response.
    """
    leaves = _sorted_leaves(subject_refs)
    target = bytes.fromhex(target_ref)
    n = len(leaves)

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if leaves[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    if lo < n and leaves[lo] == target:
        raise ValueError(f"{target_ref[:12]}... is present at index {lo}; cannot prove absence")

    def entry(i: int) -> dict:
        return {"leaf": leaves[i].hex(), "index": i,
                "path": [h.hex() for h in _audit_path(i, leaves)]}

    return {
        "tree_size": n,
        "predecessor": entry(lo - 1) if lo > 0 else None,
        "successor": entry(lo) if lo < n else None,
    }


def verify_absence(target_ref: str, proof: dict, root_hex: str) -> bool:
    """Check a non-inclusion proof against a root. Returns False, never raises.

    The adjacency check is what makes this a proof. Without it a prover can hand
    over any two leaves that happen to straddle the target -- leaving an
    unexamined gap between them that may well contain the target itself.
    """
    try:
        target = bytes.fromhex(target_ref)
        size = proof["tree_size"]
        pred, succ = proof.get("predecessor"), proof.get("successor")
    except (ValueError, TypeError, KeyError):
        return False
    if not isinstance(root_hex, str):
        return False

    if not isinstance(size, int) or size < 0:
        return False
    if size == 0:
        return pred is None and succ is None and root_hex == EMPTY_ROOT

    def check(entry) -> bytes | None:
        try:
            leaf = bytes.fromhex(entry["leaf"])
            path = [bytes.fromhex(h) for h in entry["path"]]
            index = entry["index"]
        except (ValueError, TypeError, KeyError):
            return None
        if not isinstance(index, int):
            return None
        if not verify_audit_path(leaf, index, size, path, root_hex):
            return None
        return leaf

    if pred is None:
        # Target sorts before every leaf: leaf 0 must be present and above it.
        if succ is None or succ.get("index") != 0:
            return False
        leaf = check(succ)
        return leaf is not None and target < leaf

    if succ is None:
        # Target sorts after every leaf: the last leaf must be present and below.
        if pred.get("index") != size - 1:
            return False
        leaf = check(pred)
        return leaf is not None and leaf < target

    if succ.get("index") != pred.get("index", -99) + 1:
        return False
    lo, hi = check(pred), check(succ)
    if lo is None or hi is None:
        return False
    return lo < target < hi
