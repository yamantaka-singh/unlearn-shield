"""Bearer-token auth, static scope map. No user accounts.

erasure:write callers are a consent manager or an internal job runner, not end
users hitting this API directly (per the plan); predict:invoke callers are
model-serving consumers. A static token->scope map fits that shape without a
user table, an IdP integration, or a session store this project doesn't need.
"""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import auth_tokens

_bearer = HTTPBearer(auto_error=False)


def require_scope(scope: str):
    """FastAPI dependency factory: `Depends(require_scope("erasure:write"))`."""
    def check(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
        if creds is None:
            raise HTTPException(401, "missing bearer token")
        principal = auth_tokens().get(creds.credentials)
        if principal is None:
            raise HTTPException(401, "invalid token")
        if scope not in principal["scopes"]:
            raise HTTPException(403, f"token lacks scope {scope!r}")
        return principal["principal"]
    return check
