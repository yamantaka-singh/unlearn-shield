"""Sparse Merkle tree: absence proofs that name nobody.

The sorted Merkle tree in verify/merkle.py proves absence by handing over the
target's two neighbours in sort order -- which means every certificate
discloses two other subjects' HMAC refs in cleartext. ADR 0002 recorded that as
inherent to the construction and named this module as the fix.

Here the subject_ref IS the path: bit d of the ref chooses left or right at
depth d, so every key has one fixed position among 2^256, and absence is "the
leaf at my own position is empty". The proof carries sibling *subtree hashes*
along that path. A sibling covering a populated region is a hash of that
region -- it commits to those subjects without revealing which they are.
Nobody else's identifier appears anywhere in the proof.

What still leaks: the count of non-default siblings, which is roughly the depth
at which the target's path leaves the populated part of the tree, and so hints
at tree density near that path. That is a far weaker signal than two exact
identifiers, and unlike the sorted-tree leak it does not accumulate into a
population census as an auditor collects certificates.

Not a ZK-SNARK. A SNARK would hide the sibling hashes too, at the cost of a
circom/arkworks toolchain, a trusted setup, and seconds of proving time per
erasure. Against this threat model -- an auditor who already holds the
certificate and wants to learn about other subjects -- it buys nothing an SMT
does not already provide.
"""

from hashlib import sha256

DEPTH = 256  # subject_ref is HMAC-SHA256, so the ref itself is the 256-bit path


def _leaf(key: bytes) -> bytes:
    return sha256(b"\x00" + key).digest()


def _node(left: bytes, right: bytes) -> bytes:
    return sha256(b"\x01" + left + right).digest()


# Domain-separated from both leaf and node so an empty subtree can never be
# presented as an occupied one, the same second-preimage concern that motivates
# the 0x00/0x01 prefixes in verify/merkle.py.
_EMPTY_LEAF = sha256(b"\x02unlearnshield-empty").digest()


def _default_hashes() -> list[bytes]:
    """`DEFAULTS[d]` is the root of a wholly empty subtree at depth d."""
    out = [b""] * (DEPTH + 1)
    out[DEPTH] = _EMPTY_LEAF
    for d in range(DEPTH - 1, -1, -1):
        out[d] = _node(out[d + 1], out[d + 1])
    return out


DEFAULTS = _default_hashes()
EMPTY_ROOT = DEFAULTS[0].hex()


def _bit(key: bytes, depth: int) -> int:
    return (key[depth // 8] >> (7 - depth % 8)) & 1


def _singleton_root(key: bytes, depth: int) -> bytes:
    """Root of a subtree at `depth` containing exactly `key` and nothing else."""
    node = _leaf(key)
    for d in range(DEPTH - 1, depth - 1, -1):
        sibling = DEFAULTS[d + 1]
        node = _node(node, sibling) if _bit(key, d) == 0 else _node(sibling, node)
    return node


def _subtree_root(keys: list[bytes], depth: int) -> bytes:
    # ponytail: recomputes subtrees per call, so building a root is O(n * depth)
    # hashes -- about 0.1s for a few hundred subjects. The singleton shortcut
    # above is what keeps it from being far worse, since with random keys almost
    # every key isolates within the first ~log2(n) levels. Memoise the node map
    # if a shard ever reaches six figures.
    if not keys:
        return DEFAULTS[depth]
    if len(keys) == 1:
        return _singleton_root(keys[0], depth)
    left = [k for k in keys if _bit(k, depth) == 0]
    right = [k for k in keys if _bit(k, depth) == 1]
    return _node(_subtree_root(left, depth + 1), _subtree_root(right, depth + 1))


def _keys(subject_refs) -> list[bytes]:
    return sorted({bytes.fromhex(r) for r in subject_refs})


def build_root(subject_refs) -> str:
    return _subtree_root(_keys(subject_refs), 0).hex()


def prove_absence(target_ref: str, subject_refs) -> dict:
    """Prove `target_ref` occupies an empty position.

    Raises ValueError if the key is present -- a caller asking to prove absence
    of a present subject has a bug upstream, and returning something
    proof-shaped is the worst available answer.
    """
    target = bytes.fromhex(target_ref)
    keys = _keys(subject_refs)
    if target in keys:
        raise ValueError(f"{target_ref[:12]}... is present; cannot prove absence")

    # Only siblings that differ from the all-empty default carry information;
    # the rest are recomputable by the verifier from DEFAULTS. Sending 256
    # mostly-identical hashes would work and be ~8KB of noise per certificate.
    siblings = {}
    current = keys
    for depth in range(DEPTH):
        bit = _bit(target, depth)
        same = [k for k in current if _bit(k, depth) == bit]
        other = [k for k in current if _bit(k, depth) != bit]
        sibling = _subtree_root(other, depth + 1)
        if sibling != DEFAULTS[depth + 1]:
            siblings[depth] = sibling.hex()
        current = same
        if not current:
            # Every remaining sibling is empty by construction; stop early
            # rather than walking the remaining levels to collect defaults.
            break

    return {"scheme": "smt-256", "siblings": siblings}


def verify_absence(target_ref: str, proof: dict, root_hex: str) -> bool:
    """Check an absence proof against a root. Returns False, never raises."""
    try:
        target = bytes.fromhex(target_ref)
        if len(target) * 8 != DEPTH:
            return False
        if proof.get("scheme") != "smt-256":
            return False
        raw = proof["siblings"]
        if not isinstance(raw, dict):
            return False
        siblings = {}
        for depth, value in raw.items():
            d = int(depth)
            if not 0 <= d < DEPTH:
                return False
            sibling = bytes.fromhex(value)
            if len(sibling) != 32:
                return False
            siblings[d] = sibling
    except (ValueError, TypeError, KeyError, AttributeError):
        return False

    # Start from an empty leaf: that assertion IS the claim being proved.
    node = _EMPTY_LEAF
    for depth in range(DEPTH - 1, -1, -1):
        sibling = siblings.get(depth, DEFAULTS[depth + 1])
        node = _node(node, sibling) if _bit(target, depth) == 0 else _node(sibling, node)
    return node.hex() == root_hex
