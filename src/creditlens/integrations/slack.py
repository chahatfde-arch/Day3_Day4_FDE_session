"""Approving a credit memo from Slack, where the underwriter already is.

    python -m creditlens.integrations.slack             # start the Socket Mode app
    python -m creditlens.integrations.slack --dry-run   # just print the Block Kit JSON

Socket Mode matters for a workshop: it opens a WebSocket *out* from your laptop to Slack,
so Slack never has to reach back in. No public URL, no ngrok, no firewall rules. You need
two tokens: a bot token (`xoxb-...`) and an app-level token with `connections:write`
(`xapp-...`).

The button handler does exactly what the CLI approval does: check that the person is the
Head of Credit, build an `ApprovalDecision`, call `creditlens.agents.run.resume`, and record
an `AuditEvent` with `channel="slack"`. Slack is a front door, not a second decision path.
Slack member ids are mapped to bank user ids with SLACK_USER_MAP; an unmapped clicker, or
anyone who is not head_of_credit, is refused before the paused run is touched.

Everything here imports without `slack_bolt` installed: the import happens inside the
functions that need it, so `--dry-run` and the tests work on any machine.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

from creditlens.agents import adapters
from creditlens.observability import observe
from creditlens.schemas import ApprovalDecision, AuditEvent, CreditMemo

APPROVALS_CHANNEL = os.environ.get("SLACK_APPROVALS_CHANNEL", "#credit-approvals")
APPROVE_ACTION = "creditlens_approve"
REJECT_ACTION = "creditlens_reject"


def slack_configured() -> bool:
    return bool(os.environ.get("SLACK_BOT_TOKEN") and os.environ.get("SLACK_APP_TOKEN"))


# ---------------------------------------------------------------------------
# The message
# ---------------------------------------------------------------------------


def _summary_line(memo: CreditMemo) -> str:
    exceptions = memo.policy_exceptions
    if not exceptions:
        return "No policy exceptions."
    listed = ", ".join(f"{e.rule_id}" + (f" ({e.jira_key})" if e.jira_key else "")
                       for e in exceptions)
    return f"{len(exceptions)} policy exception(s): {listed}"


def build_blocks(thread_id: str, memo: CreditMemo) -> list[dict]:
    """The Block Kit message: what the underwriter sees, and the two buttons."""
    risk = memo.risk
    risk_text = (f"{risk.risk_band} band, PD {risk.probability_of_default:.1%}"
                 if risk else "not scored")
    return [
        {"type": "header",
         "text": {"type": "plain_text", "text": f"Credit memo {memo.application_id}"}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Company*\n{memo.company_id}"},
            {"type": "mrkdwn", "text": f"*Recommendation*\n{memo.recommendation}"},
            {"type": "mrkdwn", "text": f"*Risk*\n{risk_text}"},
            {"type": "mrkdwn", "text": f"*Exceptions*\n{_summary_line(memo)}"},
        ]},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": f"*Why*\n{memo.recommendation_reasoning[:2800]}"}},
        {"type": "context", "elements": [
            {"type": "mrkdwn",
             "text": f"Thread `{thread_id}` | drafted by CreditLens AI | a human decides"}]},
        {"type": "actions", "block_id": f"creditlens::{thread_id}", "elements": [
            {"type": "button", "action_id": APPROVE_ACTION, "style": "primary",
             "text": {"type": "plain_text", "text": "Approve"}, "value": thread_id,
             "confirm": {"title": {"type": "plain_text", "text": "Approve this facility?"},
                         "text": {"type": "mrkdwn", "text": f"Application {memo.application_id}."},
                         "confirm": {"type": "plain_text", "text": "Approve"},
                         "deny": {"type": "plain_text", "text": "Cancel"}}},
            {"type": "button", "action_id": REJECT_ACTION, "style": "danger",
             "text": {"type": "plain_text", "text": "Reject"}, "value": thread_id},
        ]},
    ]


@observe(name="integrations.slack.post_for_approval")
def post_for_approval(thread_id: str, memo: CreditMemo, channel: Optional[str] = None,
                      dry_run: bool = False) -> dict:
    """Post the memo with Approve and Reject buttons. `dry_run` prints the JSON instead."""
    channel = channel or APPROVALS_CHANNEL
    blocks = build_blocks(thread_id, memo)
    payload = {"channel": channel, "text": f"Credit memo {memo.application_id} needs a decision",
               "blocks": blocks}
    if dry_run or not slack_configured():
        print(json.dumps(payload, indent=2))
        return {"posted": False, "reason": "dry run" if dry_run else "Slack is not configured",
                "payload": payload}
    from slack_sdk import WebClient

    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    response = client.chat_postMessage(**payload)
    return {"posted": True, "ts": response.get("ts"), "channel": response.get("channel")}


# ---------------------------------------------------------------------------
# The button handler
# ---------------------------------------------------------------------------


def slack_user_map() -> dict[str, str]:
    """Slack member id -> Good Bank user id, from SLACK_USER_MAP.

    Either inline JSON ({"U0ABCDEF12": "U05"}) or comma-separated pairs (U0ABCDEF12:U05).
    Slack ids and bank ids are different things: nobody is the Head of Credit just because
    they can click a button in the approvals channel.
    """
    raw = os.environ.get("SLACK_USER_MAP", "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        return {str(k): str(v) for k, v in json.loads(raw).items()}
    pairs = (item.split(":", 1) for item in raw.split(",") if ":" in item)
    return {k.strip(): v.strip() for k, v in pairs}


def handle_decision(thread_id: str, decision: str, slack_user_id: str,
                    checkpoint_path: Optional[str] = None) -> dict:
    """Resume the paused graph. Shared by both buttons, and by the tests.

    The role check happens here, before the paused run is touched, exactly as the
    command line does it: only a Slack user mapped to a head_of_credit may decide.
    """
    from creditlens.agents.run import default_checkpoint_path, pending_application_id, resume

    bank_user_id = slack_user_map().get(slack_user_id)
    try:
        user = adapters.load_user(bank_user_id) if bank_user_id else None
    except KeyError:  # mapped to an id the bank does not have
        user = None
    if user is None or user.role != "head_of_credit":
        who = f"{user.full_name} is a {user.role}" if user else f"Slack user {slack_user_id} is not mapped to a bank user"
        adapters.audit(AuditEvent(application_id="unknown", actor=slack_user_id, action="memo_decision_refused",
                                  detail=f"channel=slack; thread={thread_id}; {who}"))
        return {"status": "refused", "error": f"{who}. Policy 7.3: only the Head of Credit may decide."}

    path = checkpoint_path or default_checkpoint_path()
    application_id = pending_application_id(thread_id, checkpoint_path=path)
    if application_id is None:
        return {"status": "error", "error": f"no paused run for thread {thread_id}"}

    approval = ApprovalDecision(application_id=application_id, decided_by=user.user_id,
                                decision=decision, comment=f"decided in Slack by {slack_user_id}",
                                channel="slack")
    result = resume(thread_id, approval, checkpoint_path=path)
    adapters.audit(AuditEvent(application_id=application_id, actor=slack_user_id,
                              action=f"memo_{result['status']}",
                              detail=f"channel=slack; thread={thread_id}"))
    return result


def build_app(checkpoint_path: Optional[str] = None) -> Any:
    """The Bolt app with both button handlers registered."""
    from slack_bolt import App

    app = App(token=os.environ["SLACK_BOT_TOKEN"])

    def _respond(ack, body, say, decision: str) -> None:
        ack()
        thread_id = body["actions"][0]["value"]
        slack_user = body["user"]["id"]
        result = handle_decision(thread_id, decision, slack_user, checkpoint_path)
        if result.get("status") == "error":
            say(f"Could not record that decision: {result['error']}")
            return
        if result.get("status") == "refused":
            say(f"<@{slack_user}> that decision was refused: {result['error']}")
            return
        memo = result.get("memo")
        application = memo.application_id if memo is not None else thread_id
        say(f"<@{slack_user}> {result['status']} application {application}. "
            "The decision is in the audit trail.")

    @app.action(APPROVE_ACTION)
    def approve(ack, body, say):  # pragma: no cover - needs a live Slack connection
        _respond(ack, body, say, "approve")

    @app.action(REJECT_ACTION)
    def reject(ack, body, say):  # pragma: no cover - needs a live Slack connection
        _respond(ack, body, say, "reject")

    return app


def start_socket_mode(checkpoint_path: Optional[str] = None) -> None:  # pragma: no cover
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    app = build_app(checkpoint_path)
    print(f"CreditLens Slack app listening. Approvals go to {APPROVALS_CHANNEL}.")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _demo_memo() -> CreditMemo:
    """A tiny memo so `--dry-run` shows a real message without touching the database."""
    return CreditMemo(application_id="A001", company_id="C003",
                      applicant_summary="Demo", facility_requested="Demo",
                      financial_analysis="Demo", banking_conduct="Demo",
                      related_parties="Demo", risk_assessment="Demo",
                      recommendation="refer",
                      recommendation_reasoning="Dry run: this is what the message looks like.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m creditlens.integrations.slack",
                                     description="Approve credit memos from Slack.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the Block Kit JSON and exit, no Slack needed")
    parser.add_argument("--thread-id", default="demo-thread",
                        help="thread id to put on the buttons in a dry run")
    parser.add_argument("--application", default=None,
                        help="draft this application first and post its real memo")
    parser.add_argument("--user", default="U05", help="user id used when drafting")
    parser.add_argument("--checkpoint", default=None, help="checkpoint database")
    args = parser.parse_args(argv)

    if args.application:
        from creditlens.agents.run import default_checkpoint_path, draft_memo

        path = args.checkpoint or default_checkpoint_path()
        drafted = draft_memo(args.application, args.user, checkpoint_path=path)
        post_for_approval(drafted["thread_id"], drafted["memo"], dry_run=args.dry_run)
        return 0

    if args.dry_run:
        post_for_approval(args.thread_id, _demo_memo(), dry_run=True)
        return 0

    if not slack_configured():
        print("SLACK_BOT_TOKEN and SLACK_APP_TOKEN are not set.")
        print("Approve from the command line instead:")
        print("  python -m creditlens.agents.approve <thread_id> approve --user U05")
        return 2

    start_socket_mode(args.checkpoint)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
