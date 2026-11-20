"""Start the LiteLLM proxy from `litellm_config.yaml`.

Why a proxy at all, when `creditlens.gateway.llm` already falls back? Because the in-process
gateway protects one process. The proxy protects a room: one place holds the keys, one place
holds the rate limits and the budget, and every laptop points `LLM_PRIMARY` at it. On Day 7
it is also the thing we deliberately break, to watch the fallbacks work.

    python -m creditlens.gateway.proxy                # start on http://localhost:4000
    python -m creditlens.gateway.proxy --port 4100
    python -m creditlens.gateway.proxy --print-config # just show the file and exit

Nothing here is required for tests: without the proxy extra installed it prints the install
command and exits cleanly.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path(__file__).with_name("litellm_config.yaml")
INSTALL_HINT = 'pip install "litellm[proxy]>=1.50"'


def proxy_available() -> bool:
    """True when the `litellm` command exists and the proxy extra is importable."""
    if shutil.which("litellm") is None:
        return False
    try:
        import litellm.proxy  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def build_command(port: int, host: str, detailed_debug: bool = False) -> list[str]:
    cmd = ["litellm", "--config", str(CONFIG_PATH), "--port", str(port), "--host", host]
    if detailed_debug:
        cmd.append("--detailed_debug")
    return cmd


def start(port: int = 4000, host: str = "127.0.0.1", detailed_debug: bool = False) -> int:
    """Run the proxy in the foreground. Ctrl-C stops it."""
    if not CONFIG_PATH.exists():
        print(f"Config not found: {CONFIG_PATH}")
        return 2
    if not proxy_available():
        print("The LiteLLM proxy is not installed in this environment.")
        print(f"Install it with:  {INSTALL_HINT}")
        print(f"Then point the project at it:  LLM_PRIMARY=openai/creditlens-primary and "
              f"OPENAI_API_BASE=http://{host}:{port}")
        return 1

    cmd = build_command(port, host, detailed_debug)
    print("Starting the LiteLLM proxy:")
    print("  " + " ".join(cmd))
    print(f"  Health check:  curl http://{host}:{port}/health")
    print("  Stop it with Ctrl-C.")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0
    except FileNotFoundError:
        print(f"Could not launch `litellm`. Install it with:  {INSTALL_HINT}")
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LiteLLM proxy for CreditLens AI")
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--detailed-debug", action="store_true")
    parser.add_argument("--print-config", action="store_true", help="print the YAML and exit")
    args = parser.parse_args(argv)

    if args.print_config:
        print(CONFIG_PATH.read_text(encoding="utf-8"))
        return 0
    return start(port=args.port, host=args.host, detailed_debug=args.detailed_debug)


if __name__ == "__main__":
    sys.exit(main())
