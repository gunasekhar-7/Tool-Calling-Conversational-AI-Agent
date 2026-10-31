"""Streamlit front end for the support agent.

The dashboard never decides anything itself: every write is staged by the API and
approved here by a human, and the API remains the only thing that touches data.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import streamlit as st

from api_client import AgentAPI, ApiError

st.set_page_config(
    page_title="Order support agent",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="expanded",
)

EXAMPLES = [
    ("Where is my order?", "Where is ORD-1002 right now?"),
    ("Something arrived damaged", "The cable in ORD-1001 arrived bent. Please open a ticket."),
    ("Am I still inside the refund window?", "Can I still get a refund on ORD-1003?"),
]

PRIORITY_COLOUR = {"high": "#b42318", "normal": "#475467", "low": "#667085"}


def _setting(name: str, default: str = "") -> str:
    # A fresh checkout has no .streamlit/secrets.toml; asking for it would raise.
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except FileNotFoundError:
        pass
    return os.environ.get(name, default)


def _keyring() -> dict[str, str]:
    """label -> token, so the picker never shows a token to the person using it."""
    raw = _setting("CUSTOMER_TOKENS")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(label): str(token) for label, token in parsed.items()}


def api() -> AgentAPI | None:
    base = st.session_state.get("api_base") or _setting("API_BASE", "http://127.0.0.1:8000")
    keyring = _keyring()
    if keyring:
        label = st.session_state.get("customer_label") or next(iter(keyring))
        key = keyring.get(label, "")
    else:
        key = st.session_state.get("api_key") or os.environ.get("API_TOKEN", "")
    if not key:
        return None
    return AgentAPI(base, key)


def _reset_session() -> None:
    for key in ("session_id", "messages", "pending", "turn_error", "resolved"):
        st.session_state.pop(key, None)
    _state_defaults()


def _state_defaults() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("resolved", None)
    st.session_state.setdefault("turn_error", None)


def _render_tools(tools: list[str]) -> None:
    if not tools:
        return
    chips = " ".join(f"<span class='tool-chip'>{name}</span>" for name in dict.fromkeys(tools))
    st.markdown(chips, unsafe_allow_html=True)


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"] or "_…_")
            _render_tools(message.get("tools") or [])
            if message.get("warning"):
                st.caption(f"⚠️ {message['warning']}")


def _approval_card(pending: dict[str, Any], client: AgentAPI) -> None:
    """The human-in-the-loop gate. Nothing here runs until a button is pressed."""
    with st.container(border=True):
        st.markdown("#### Confirm before I create this ticket")
        st.caption(
            "Nothing has been written yet. Approving creates exactly one ticket for your "
            "account; rejecting leaves your data untouched."
        )
        left, right = st.columns([2, 1])
        with left:
            st.markdown(f"**Subject** · {pending.get('subject', '—')}")
            st.markdown(pending.get("body") or "_No description_")
            order_id = pending.get("order_id")
            st.markdown(f"**Linked order** · {order_id or 'none'}")
        with right:
            priority = str(pending.get("priority", "normal"))
            st.markdown(
                f"<span class='priority' style='color:{PRIORITY_COLOUR.get(priority, '#475467')}'>"
                f"{priority.upper()} priority</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Expires {str(pending.get('expires_at', ''))[:16].replace('T', ' ')} UTC")

        approve, reject = st.columns(2)
        decision = None
        if approve.button("Approve and create", type="primary", use_container_width=True):
            decision = "approve"
        if reject.button("Reject", use_container_width=True):
            decision = "reject"

    if decision:
        _resolve(pending, decision, client)


def _resolve(pending: dict[str, Any], decision: str, client: AgentAPI) -> None:
    session_id = st.session_state.session_id
    key = st.session_state.pending.setdefault("idempotency_key", uuid.uuid4().hex)
    try:
        with st.spinner("Recording your decision…"):
            result = client.decide(
                session_id,
                pending_action_id=pending["pending_action_id"],
                decision=decision,
                idempotency_key=key,
                preview_hash=pending["preview_hash"],
            )
    except ApiError as exc:
        st.error(f"Your decision was not recorded: {exc.user_message()}")
        if exc.code in {"conflict", "approval_expired", "approval_mismatch"}:
            st.session_state.pending = None
            st.rerun()
        return

    st.session_state.pending = None
    st.session_state.resolved = {"decision": decision, **result}
    reply = result.get("reply") or (
        "The ticket is created."
        if result.get("status") == "executed"
        else "Understood — I won't create that ticket."
    )
    for index in range(len(st.session_state.messages) - 1, -1, -1):
        message = st.session_state.messages[index]
        if message.get("role") == "assistant" and message.get("content") == (
            "_Waiting for your approval._"
        ):
            st.session_state.messages[index] = {
                "role": "assistant",
                "content": reply,
                "tools": [],
                "warning": None,
            }
            break
    else:
        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "tools": [], "warning": None}
        )
    st.rerun()


def _run_turn(client: AgentAPI, session_id: str, text: str) -> None:
    st.session_state.messages.append(
        {"role": "user", "content": text, "tools": [], "warning": None}
    )
    with st.chat_message("user"):
        st.markdown(text)

    tools: list[str] = []
    assistant_text = ""
    warning: str | None = None
    pending: dict[str, Any] | None = None

    with st.chat_message("assistant"):
        tool_slot = st.empty()
        text_slot = st.empty()
        text_slot.markdown("_Thinking…_")
        try:
            for event in client.stream_message(session_id, text):
                if event.name == "token":
                    assistant_text += str(event.data.get("text", ""))
                    text_slot.markdown(assistant_text)
                elif event.name == "tool_call":
                    name = str(event.data.get("name", ""))
                    if name and name not in tools:
                        tools.append(name)
                        with tool_slot.container():
                            _render_tools(tools)
                elif event.name == "confirmation_required":
                    pending = event.data
                elif event.name == "grounding_warning":
                    unsupported = ", ".join(event.data.get("unsupported") or [])
                    warning = f"I left out details I couldn't verify: {unsupported}"
                elif event.name == "replace":
                    assistant_text = str(event.data.get("text", assistant_text))
                    text_slot.markdown(assistant_text)
                elif event.name == "done":
                    assistant_text = str(event.data.get("text", assistant_text))
                elif event.name == "error":
                    code = event.data.get("code", "internal_error")
                    message = event.data.get("message", "Something went wrong.")
                    st.session_state.turn_error = (code, message, event.data.get("request_id"))
                    text_slot.empty()
                    return
        except ApiError as exc:
            st.session_state.turn_error = (exc.code, exc.message, exc.request_id)
            text_slot.empty()
            return
        except Exception as exc:  # network layer: never leave the user guessing
            st.session_state.turn_error = ("connection_failed", str(exc), None)
            text_slot.empty()
            return

    if pending:
        st.session_state.pending = pending
        st.session_state.messages.append(
            {"role": "assistant", "content": "_Waiting for your approval._", "tools": tools}
        )
        st.rerun()

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_text, "tools": tools, "warning": warning}
    )
    st.rerun()


_state_defaults()

st.markdown(
    """
    <style>
      .tool-chip {
        display:inline-block; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
        font-size:0.72rem; padding:0.15rem 0.5rem; margin:0.35rem 0.3rem 0 0;
        border-radius:999px; background:#eef2f7; color:#344054; border:1px solid #d0d5dd;
      }
      .priority { font-weight:600; letter-spacing:0.04em; font-size:0.78rem; }
      .lede { color:#475467; font-size:1.02rem; margin-top:-0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Connection")
    st.text_input("API base", key="api_base", value=_setting("API_BASE", "http://127.0.0.1:8000"))

    keyring = _keyring()
    if keyring:
        st.selectbox(
            "Signed in as",
            list(keyring),
            key="customer_label",
            help=(
                "Each account has its own token. Switching accounts switches the data "
                "the agent can see — it has no way to reach anyone else's."
            ),
        )
    else:
        st.text_input(
            "API token",
            key="api_key",
            type="password",
            help=(
                "One token per customer. Set CUSTOMER_TOKENS to get an account picker "
                "instead of pasting a token."
            ),
        )

    client = api()
    if client is None:
        st.warning("Add a customer token to start a conversation.")
        st.stop()
    assert client is not None  # st.stop() ends the run

    try:
        health = client.health()
        st.success(f"API online · v{health.get('version', '?')}", icon="🟢")
        customers = client.customers()
    except ApiError as exc:
        st.error(f"Can't reach the API: {exc.user_message()}")
        st.stop()
    except Exception as exc:
        st.error(f"Can't reach {st.session_state.api_base}: {exc}")
        st.stop()

    if not customers:
        st.error("This token isn't linked to a customer yet.")
        st.stop()
    me = customers[0]
    chosen = str(me["name"])
    st.caption(f"Account · {me.get('email', chosen)}")
    st.caption("The agent only ever sees data for this account.")

    if st.button("New conversation", use_container_width=True):
        _reset_session()
        st.rerun()

    if st.session_state.get("session_id"):
        st.caption(f"Session `{st.session_state.session_id[:8]}`")

st.title("Order support agent")
st.markdown(
    "<p class='lede'>Ask about an order, a refund window, or a support ticket. "
    "Anything that changes your account waits for your explicit approval.</p>",
    unsafe_allow_html=True,
)

if st.session_state.get("customer_name") != chosen:
    _reset_session()
    st.session_state.customer_name = chosen

if not st.session_state.get("session_id"):
    try:
        st.session_state.session_id = client.create_session()
    except ApiError as exc:
        st.error(f"Couldn't start a session: {exc.user_message()}")
        st.stop()

if error := st.session_state.get("turn_error"):
    code, message, request_id = error
    hint = {
        "rate_limited": "Give it a moment, then send the message again.",
        "model_unavailable": "The model key isn't configured on the API.",
        "policy_denied": "Rephrase the request as a normal support question.",
        "approval_pending": "Approve or reject the ticket below before asking something else.",
        "connection_failed": "Check that the API is running and the base URL is right.",
    }.get(code, "Try again — if it keeps failing, quote the request id.")
    st.error(f"{message} {hint}" + (f" (request {request_id})" if request_id else ""))
    st.session_state.turn_error = None

if not st.session_state.messages and not st.session_state.get("pending"):
    st.markdown("#### Try one of these")
    columns = st.columns(len(EXAMPLES))
    for column, (label, prompt) in zip(columns, EXAMPLES, strict=True):
        if column.button(label, use_container_width=True):
            _run_turn(client, st.session_state.session_id, prompt)

_render_history()

if pending := st.session_state.get("pending"):
    _approval_card(pending, client)
elif resolved := st.session_state.get("resolved"):
    if resolved.get("decision") == "approve" and resolved.get("ticket_id"):
        st.success(f"Ticket {resolved['ticket_id']} is open. Support will follow up by email.")
    elif resolved.get("decision") == "reject":
        st.info("Rejected. Nothing was written to your account.")
    st.session_state.resolved = None

if message := st.chat_input(
    "Ask about an order, refund or ticket…",
    disabled=bool(st.session_state.get("pending")),
):
    if not st.session_state.get("session_id"):
        st.error("Start a conversation first.")
    else:
        _run_turn(client, st.session_state.session_id, message)
