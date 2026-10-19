"""Callers: admins by API key (X-API-Key), users by a JWT bearer token with a tenant claim.

A user's tenant and ACL groups come only from the token (claims `tenant_id`, `groups`),
so a request can never widen its own retrieval scope.
"""
import hmac
from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException, Request, status

from selfrag.api.settings import ApiSettings


@dataclass(frozen=True)
class Principal:
    tenant_id: str | None               # None for admins: they choose the tenant per request
    acl_groups: tuple[str, ...] = ()
    is_admin: bool = False


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail,
                         headers={"WWW-Authenticate": "Bearer"})


def authenticate(request: Request, x_api_key: str | None = Header(None),
                 authorization: str | None = Header(None)) -> Principal:
    cfg: ApiSettings = request.app.state.api_settings
    if x_api_key is not None:
        if cfg.admin_api_key and hmac.compare_digest(x_api_key.encode(),
                                                     cfg.admin_api_key.encode()):
            return Principal(tenant_id=None, is_admin=True)
        raise _unauthorized("invalid API key")

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise _unauthorized("missing credentials")
    if not cfg.jwt_secret:
        raise _unauthorized("token auth is not configured")
    try:
        claims = jwt.decode(token, cfg.jwt_secret, algorithms=[cfg.jwt_algorithm],
                            options={"require": ["exp", "tenant_id"]})
    except jwt.PyJWTError:
        raise _unauthorized("invalid or expired token")
    tenant, groups = claims["tenant_id"], claims.get("groups", [])
    if not isinstance(tenant, str) or not tenant or not isinstance(groups, list) \
            or not all(isinstance(g, str) for g in groups):
        raise _unauthorized("malformed token claims")
    return Principal(tenant_id=tenant, acl_groups=tuple(groups))


def require_admin(who: Principal = Depends(authenticate)) -> Principal:
    if not who.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    return who
