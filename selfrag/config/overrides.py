"""Per-request config overrides (admin only), validated against allowed ranges."""
from dataclasses import replace
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class OverridesNotAllowed(PermissionError):
    """Raised when a non-admin caller sends overrides (the API maps it to 403)."""


class ConfigOverrides(BaseModel):
    """Out-of-range or unknown fields raise pydantic.ValidationError (the API maps it to 400)."""
    model_config = ConfigDict(extra="forbid")

    delta: float | None = Field(None, ge=0.0, le=1.0)      # retrieval threshold
    k: int | None = Field(None, ge=1, le=20)                # passages per step
    B: int | None = Field(None, ge=1, le=4)                 # beam width
    w_rel: float | None = Field(None, ge=0.0, le=5.0)
    w_sup: float | None = Field(None, ge=0.0, le=5.0)
    w_use: float | None = Field(None, ge=0.0, le=5.0)
    critic_mode: Literal["logprobs", "sample", "verbal"] | None = None


def apply_overrides(cfg: T, overrides: ConfigOverrides | None, *, is_admin: bool) -> T:
    """Return a copy of the (frozen dataclass) config with the overridden fields replaced."""
    changes = overrides.model_dump(exclude_none=True) if overrides else {}
    if changes and not is_admin:
        raise OverridesNotAllowed("config overrides are admin only")
    return replace(cfg, **changes)
