"""The no-Slack approval path: a human underwriter decides from the terminal.

    python -m creditlens.agents.approve <thread_id> approve --user U05
    python -m creditlens.agents.approve <thread_id> reject  --user U05 --comment "DSCR too thin"
    python -m creditlens.agents.approve <thread_id> approve --user U05 --amount 15000   # a counter-offer

Only the Head of Credit may approve (policy 7.3), so the command checks the caller's role
before it resumes the graph.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from creditlens.agents import adapters
from creditlens.agents.run import default_checkpoint_path, pending_application_id, resume
from creditlens.schemas import ApprovalDecision


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m creditlens.agents.approve",
        description="Approve or reject a credit memo that is waiting at the underwriter gate.")
    parser.add_argument("thread_id", help="the thread id printed by `python -m creditlens.agents.run`")
    parser.add_argument("decision", choices=["approve", "reject"])
    parser.add_argument("--user", default="U05", help="user id of the underwriter")
    parser.add_argument("--comment", default="", help="why")
    parser.add_argument("--amount", type=int, default=None,
                        help="approved amount in INR thousand, when it differs from the request")
    parser.add_argument("--checkpoint", default=None, help="checkpoint database")
    args = parser.parse_args(argv)

    path = args.checkpoint or default_checkpoint_path()
    user = adapters.load_user(args.user)
    if user.role != "head_of_credit":
        print(f"{user.full_name} is a {user.role}. Policy 7.3: only the Head of Credit may decide.")
        return 2

    application_id = pending_application_id(args.thread_id, checkpoint_path=path)
    if application_id is None:
        print(f"No paused run found for thread {args.thread_id}.")
        return 1

    decision = ApprovalDecision(application_id=application_id, decided_by=user.user_id,
                                decision=args.decision, comment=args.comment,
                                approved_inr_000=args.amount, channel="cli")
    result = resume(args.thread_id, decision, checkpoint_path=path)
    print(f"Application {application_id} is now {result['status']} "
          f"(decided by {user.full_name} from the command line).")
    memo = result.get("memo")
    if memo is not None:
        print(f"Recommendation on file: {memo.recommendation}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
