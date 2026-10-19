"""Request bodies. Responses reuse AnswerResponse (selfrag.models) and Job (selfrag.api.jobs)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from selfrag.config.overrides import ConfigOverrides


class AnswerFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: str | None = None
    tenant_id: str | None = None                # admin only; users get it from their token
    acl_groups: list[str] | None = None         # admin only; users get it from their token


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["sentence", "answer"] | None = None      # default: segment_unit setting
    filters: AnswerFilters = AnswerFilters()
    overrides: ConfigOverrides | None = None                # admin only
    debug: bool = False                     # admin only: add the decision trace to the response
