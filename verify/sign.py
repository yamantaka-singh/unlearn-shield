"""Ed25519 signing over the canonical manifest bytes.

The private key never lives in this repo. It comes from UNLEARNSHIELD_SIGNING_KEY
(hex seed) or a secrets manager. The public key is checked in at
verify/public_key.hex so the standalone verifier ships with what it needs.
"""

import os

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey

from verify.manifest import canonical_bytes

PUBLIC_KEY_PATH = os.path.join(os.path.dirname(__file__), "public_key.hex")


def signing_key() -> SigningKey:
    seed = os.environ.get("UNLEARNSHIELD_SIGNING_KEY")
    if not seed:
        raise RuntimeError(
            "UNLEARNSHIELD_SIGNING_KEY unset. Generate a dev key with:\n"
            "  python -m verify.sign --generate")
    return SigningKey(bytes.fromhex(seed))


def sign_manifest(manifest: dict) -> str:
    return signing_key().sign(canonical_bytes(manifest)).signature.hex()


def load_public_key(path: str = PUBLIC_KEY_PATH) -> VerifyKey:
    with open(path) as f:
        return VerifyKey(bytes.fromhex(f.read().strip()))


def verify_manifest(manifest: dict, signature_hex: str, public_key: VerifyKey) -> bool:
    """Re-canonicalises before checking, so a tampered field changes the bytes."""
    try:
        public_key.verify(canonical_bytes(manifest), bytes.fromhex(signature_hex))
        return True
    except (BadSignatureError, ValueError, TypeError):
        return False


def _generate_dev_key(private_path: str = ".signing_key") -> None:
    """Dev only. Writes the seed to a gitignored file at 0600 rather than
    printing it -- stdout ends up in shell history and CI logs, and a key that
    leaked once is a key that has to be rotated everywhere it signed."""
    key = SigningKey.generate()
    with open(PUBLIC_KEY_PATH, "w") as f:
        f.write(key.verify_key.encode().hex() + "\n")
    fd = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(bytes(key).hex())
    print(f"public key  -> {PUBLIC_KEY_PATH} (commit this)")
    print(f"private key -> {private_path} (gitignored, never commit)")
    print(f"use it with: export UNLEARNSHIELD_SIGNING_KEY=$(cat {private_path})")


if __name__ == "__main__":
    _generate_dev_key()
