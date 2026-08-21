"""Standalone erasure certificate verifier.

    python -m verify.verifier_cli manifest.json [--public-key verify/public_key.hex]

This file, and everything it imports, must stay free of engine/, gateway/,
worker/, config/ and the database. If verifying a certificate requires access to
the system that produced it, it is not a proof -- it is the operator asserting
their own compliance and handing you a script that agrees. An auditor gets this
directory, a manifest, and nothing else.

`tests/unit/test_verifier_isolation.py` enforces that by copying verify/ into a
bare directory and running it there.
"""

import argparse
import json
import sys

from nacl.signing import VerifyKey

from verify import merkle, smt
from verify.manifest import REQUIRED_FIELDS, canonical_bytes


def verify_absence(subject_ref: str, proof: dict, dataset_root: str) -> tuple[bool, str]:
    """Dispatch on the proof's declared scheme, and say which one ran.

    A certificate outlives the code that issued it, so this keeps checking
    sorted-Merkle proofs (verify/merkle.py) issued before the sparse tree
    replaced it. New certificates are 'smt-256', which names no other subject
    -- see verify/smt.py.
    """
    if isinstance(proof, dict) and proof.get("scheme") == "smt-256":
        return smt.verify_absence(subject_ref, proof, dataset_root), "sparse Merkle (smt-256)"
    return merkle.verify_absence(subject_ref, proof, dataset_root), "sorted Merkle (legacy)"


def verify_certificate(manifest: dict, public_key: VerifyKey) -> tuple[bool, list[str]]:
    """Returns (ok, findings). Findings are ordered most-fundamental first."""
    findings = []

    missing = [f for f in REQUIRED_FIELDS if f not in manifest]
    if missing:
        return False, [f"manifest is missing required fields: {missing}"]

    signature = manifest.pop("signature", None)
    if not signature:
        return False, ["manifest carries no signature"]

    try:
        public_key.verify(canonical_bytes(manifest), bytes.fromhex(signature))
        findings.append("signature valid (Ed25519)")
    except Exception:
        return False, ["SIGNATURE INVALID -- manifest was altered or signed by another key"]

    # Only meaningful after the signature checks out: before that, dataset_root
    # and absence_proof are attacker-controlled and proving them consistent
    # with each other proves nothing.
    proof_ok, scheme = verify_absence(manifest["subject_ref"], manifest["absence_proof"],
                                      manifest["dataset_root"])
    if proof_ok:
        findings.append(f"absence proof valid against dataset_root "
                        f"{manifest['dataset_root'][:16]}... [{scheme}]")
    else:
        return False, findings + ["ABSENCE PROOF INVALID -- subject is not provably "
                                  "absent from the retained set"]

    findings.append(f"subject {manifest['subject_ref'][:16]}... absent from shard "
                    f"{manifest['shard']} at model_version {manifest['model_version']}")
    findings.append(f"weights {manifest['result_weights'][:16]}... produced by "
                    f"code_digest {manifest['code_digest']}")
    return True, findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", help="path to a signed manifest JSON file")
    parser.add_argument("--public-key", default=None)
    args = parser.parse_args(argv)

    with open(args.manifest) as f:
        manifest = json.load(f)

    key_path = args.public_key or __file__.rsplit("/", 1)[0] + "/public_key.hex"
    with open(key_path) as f:
        public_key = VerifyKey(bytes.fromhex(f.read().strip()))

    ok, findings = verify_certificate(manifest, public_key)
    for line in findings:
        print(("  ok  " if ok else "  --  ") + line)
    print("VERIFIED" if ok else "REJECTED")

    # What this does and does not establish. An auditor reading only the happy
    # path should not leave thinking more was proven than was.
    if ok:
        print("\nProves: the subject is absent from the record set whose Merkle root\n"
              "        is named here, and this manifest was signed by the holder of\n"
              "        the private key.\n"
              "Does not prove: that these weights were trained on that record set.\n"
              "        Nothing binds result_weights to dataset_root. That link rests\n"
              "        on code_digest and on re-running sampled rebuilds.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
