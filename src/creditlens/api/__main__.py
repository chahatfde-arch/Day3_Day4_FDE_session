"""Run the API: `python -m creditlens.api`.

    python -m creditlens.api                 # http://localhost:8000, docs at /docs
    python -m creditlens.api --reload        # restart on every code change
    python -m creditlens.api --port 8080

Then, in another terminal:

    curl http://localhost:8000/health
    curl -H "X-User-Id: U05" http://localhost:8000/applications
"""
from __future__ import annotations

import argparse

from creditlens.api.app import create_app
from creditlens.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CreditLens AI API")
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 to accept connections from outside the laptop")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart when a source file changes")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    import uvicorn

    print(f"CreditLens AI API on http://{args.host}:{args.port}  (docs at /docs)")
    print(f"database: {settings.database_url}")
    if args.reload:
        # --reload needs an import string, because the worker process re-imports the app.
        uvicorn.run("creditlens.api.app:create_app", factory=True, host=args.host,
                    port=args.port, reload=True, log_level=args.log_level)
    else:
        uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
