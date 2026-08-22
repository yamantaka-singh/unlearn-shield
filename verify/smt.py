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
    if not keys:
        return DEFAULTS[depth]
    if len(keys) == 1:
        return _singleton_root(keys[0], depth)
    left = [k for k in keys if _bit(k, depth) == 0]
    right = [k for k in keys if _bit(k, depth) == 1]
    return _node(_subtree_root(left, depth + 1), _subtree_root(right, depth + 1))


def _keys(subject_refs) -> list[bytes]:
    return sorted({bytes.fromhex(r) for r in subject_refs})


# A materialised node: (hash, left, right, singleton_key).
#   internal  -> (h, left, right, None)
#   singleton -> (h, None, None, key)      one key occupies this whole subtree
#   empty     -> (DEFAULTS[depth], None, None, None)
def _build(keys: list[bytes], depth: int):
    if not keys:
        return (DEFAULTS[depth], None, None, None)
    if len(keys) == 1:
        return (_singleton_root(keys[0], depth), None, None, keys[0])
    left = _build([k for k in keys if _bit(k, depth) == 0], depth + 1)
    right = _build([k for k in keys if _bit(k, depth) == 1], depth + 1)
    return (_node(left[0], right[0]), left, right, None)


class SparseMerkleTree:
    """Build the tree once; serve the root and every absence proof from it.

    This exists because the free functions below rebuild the tree from scratch
    on every call, and a rebuild issues one `build_root` plus one
    `prove_absence` per erased subject -- so a batch paid for the whole tree
    (n+1) times. Measured on 276,075 subjects, which is a realistic shard at
    the scale this project targets:

        build_root      40.7s
        prove_absence   40.8s     <- a second full traversal, same work again
        SparseMerkleTree(refs)    one build, root and proofs then ~free

    The per-key floor is inherent and worth naming so nobody hunts for a
    bigger win that is not there: a key isolates from its neighbours at around
    depth log2(n) (~18 here), but the path from its leaf at depth 256 up to
    that point is still ~238 hashes of "combine with an empty sibling". At
    276k keys that is ~65M SHA-256 calls, which is the whole of the remaining
    build cost. Shortening DEPTH would cut it proportionally and is
    deliberately NOT done: the 256-bit path is what makes a subject's position
    unforgeable, and trading that for wall-clock is a security change wearing
    a performance costume.
    """

    def __init__(self, subject_refs):
        self._keys = _keys(subject_refs)
        self._present = set(self._keys)
        self._root = _build(self._keys, 0)

    @property
    def root(self) -> str:
        return self._root[0].hex()

    def prove_absence(self, target_ref: str) -> dict:
        """Walk down the built tree collecting sibling subtree hashes.

        Only siblings that differ from the all-empty default carry
        information; the rest are recomputable by the verifier from DEFAULTS.
        Sending 256 mostly-identical hashes would work and be ~8KB of noise
        per certificate.
        """
        target = bytes.fromhex(target_ref)
        if target in self._present:
            raise ValueError(f"{target_ref[:12]}... is present; cannot prove absence")

        siblings = {}
        node = self._root
        for depth in range(DEPTH):
            _, left, right, key = node
            if left is not None:
                bit = _bit(target, depth)
                same, other = (left, right) if bit == 0 else (right, left)
                if other[0] != DEFAULTS[depth + 1]:
                    siblings[depth] = other[0].hex()
                node = same
                continue
            if key is None:
                # Empty subtree: every remaining sibling is a default, and the
                # verifier fills those in itself.
                break
            # A collapsed singleton. The one key here shares the target's
            # prefix down to wherever their bits first differ; every level
            # above that has an empty sibling and contributes nothing.
            for d in range(depth, DEPTH):
                if _bit(target, d) != _bit(key, d):
                    siblings[d] = _singleton_root(key, d + 1).hex()
                    break
            break

        return {"scheme": "smt-256", "siblings": siblings}


def build_root(subject_refs) -> str:
    return _subtree_root(_keys(subject_refs), 0).hex()


def prove_absence(target_ref: str, subject_refs) -> dict:
    """Single-shot absence proof. Kept for callers proving one thing against a
    set they hold once; a rebuild should use SparseMerkleTree instead so the
    root and the proofs share one build."""
    return SparseMerkleTree(subject_refs).prove_absence(target_ref)


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
