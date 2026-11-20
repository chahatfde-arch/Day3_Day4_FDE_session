"""A SOAP service that behaves like a legacy one: slow on request, flaky on request.

    python -m creditlens.integrations.soap_server --port 8001
    python -m creditlens.integrations.soap_server --port 8001 --slow 3 --flaky 0.5

`--slow 3` makes every call take three seconds, so a five-second timeout is the difference
between a working agent and a hung one. `--flaky 0.5` fails half the calls, so the retry
and the circuit breaker in `soap.py` have something to do.

The service is `RepaymentHistory` with one method, `get_history(customer_id)`, returning a
list of instalments. It is built with spyne and served by `wsgiref`, which is in the Python
standard library, so there is nothing to install beyond spyne and nothing to run in Docker.
"""
from __future__ import annotations

import argparse
import logging
import random
import time
from typing import Any

from creditlens.integrations.soap import generate_history

TARGET_NAMESPACE = "creditlens.goodbank.legacy"

#: Set from the command line and read inside the service method.
SETTINGS: dict[str, float] = {"slow": 0.0, "flaky": 0.0}


def build_application() -> Any:
    """Build the spyne application. Imported lazily so the module imports without spyne."""
    from spyne import Application, Decimal, Integer, Iterable, ServiceBase, Unicode, rpc
    from spyne.model.complex import ComplexModel
    from spyne.protocol.soap import Soap11

    class RepaymentRecord(ComplexModel):
        __namespace__ = TARGET_NAMESPACE
        due_date = Unicode
        amount = Decimal
        paid_on = Unicode
        days_late = Integer

    class RepaymentHistoryService(ServiceBase):
        """The legacy interface, warts and all."""

        @rpc(Unicode, _returns=Iterable(RepaymentRecord))
        def get_history(ctx, customer_id):
            if SETTINGS["slow"]:
                time.sleep(SETTINGS["slow"])
            if SETTINGS["flaky"] and random.random() < SETTINGS["flaky"]:
                raise ValueError("CORE-0500: downstream system temporarily unavailable")
            for row in generate_history(customer_id or ""):
                yield RepaymentRecord(due_date=row["due_date"], amount=row["amount"],
                                      paid_on=row["paid_on"], days_late=row["days_late"])

    return Application([RepaymentHistoryService], tns=TARGET_NAMESPACE,
                       name="RepaymentHistory", in_protocol=Soap11(validator="lxml"),
                       out_protocol=Soap11())


def build_wsgi_app() -> Any:
    from spyne.server.wsgi import WsgiApplication

    return WsgiApplication(build_application())


def serve(port: int = 8001, slow: float = 0.0, flaky: float = 0.0, host: str = "127.0.0.1") -> None:
    """Run the service until Ctrl+C."""
    from wsgiref.simple_server import make_server

    SETTINGS["slow"] = float(slow)
    SETTINGS["flaky"] = float(flaky)
    logging.basicConfig(level=logging.WARNING)
    server = make_server(host, port, build_wsgi_app())
    print(f"RepaymentHistory SOAP service on http://{host}:{port}")
    print(f"  WSDL:  http://{host}:{port}/?wsdl")
    print(f"  slow={slow}s  flaky={flaky}")
    print("Stop with Ctrl+C.")
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m creditlens.integrations.soap_server",
                                     description="A deliberately legacy SOAP service.")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--slow", type=float, default=0.0,
                        help="seconds to sleep before answering, for example 3")
    parser.add_argument("--flaky", type=float, default=0.0,
                        help="probability of failing a call, for example 0.5")
    args = parser.parse_args(argv)
    serve(port=args.port, slow=args.slow, flaky=args.flaky, host=args.host)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
