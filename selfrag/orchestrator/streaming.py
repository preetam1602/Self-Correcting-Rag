"""Streaming events: segments are sent once their beam leads at the end of a step.

If the leading beam changes, a beam_switch tells the client to drop its text from
`from_index` on, and the new leader's segments follow.
"""
from dataclasses import dataclass

from pydantic import BaseModel

from selfrag.models import AnswerResponse, AnswerSegment, Citation


class SegmentEvent(BaseModel):
    index: int
    text: str
    citation: Citation | None


class BeamSwitchEvent(BaseModel):
    from_index: int


class ErrorEvent(BaseModel):
    trace_id: str
    message: str


@dataclass(frozen=True)
class StreamEvent:
    event: str          # "segment" | "beam_switch" | "done" | "error"
    data: SegmentEvent | BeamSwitchEvent | AnswerResponse | ErrorEvent


def sync_leader(sent: list[AnswerSegment], leader: list[AnswerSegment]) -> list[StreamEvent]:
    """Events that bring the client from `sent` to `leader`; updates `sent` in place."""
    common = 0
    while common < min(len(sent), len(leader)) and sent[common] == leader[common]:
        common += 1
    events = []
    if common < len(sent):
        events.append(StreamEvent("beam_switch", BeamSwitchEvent(from_index=common)))
        del sent[common:]
    for i in range(common, len(leader)):
        events.append(StreamEvent("segment", SegmentEvent(index=i, text=leader[i].text,
                                                          citation=leader[i].citation)))
        sent.append(leader[i])
    return events
