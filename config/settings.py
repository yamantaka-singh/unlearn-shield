"""Env-driven config. One place, no defaults that differ between dev and prod."""

import os

SEED = int(os.environ.get("UNLEARNSHIELD_SEED", "1337"))
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://unlearnshield:unlearnshield@localhost:5432/unlearnshield")

# Identifies the exact image weights were produced by. Determinism is only
# asserted within one code_digest -- see config/determinism.py.
CODE_DIGEST = os.environ.get("CODE_DIGEST", "dev-unpinned")
