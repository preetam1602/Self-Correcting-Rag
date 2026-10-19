"""Run the API server.

    python -m selfrag.api --host 127.0.0.1 --port 8000
"""
import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m selfrag.api",
                                     description="Serve the Self-RAG HTTP API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("selfrag.api.app:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
