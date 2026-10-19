"""Server-Sent Events wire format."""
from selfrag.orchestrator.streaming import StreamEvent


def encode(event: StreamEvent) -> str:
    return f"event: {event.event}\ndata: {event.data.model_dump_json()}\n\n"
