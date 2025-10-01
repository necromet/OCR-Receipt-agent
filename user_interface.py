import base64
import json
import re
from typing import List, Dict, Any

import streamlit as st

# Use OpenAIService from utils
from utils.openai_service import OpenAIService
from receipt_parsing import receipt_parsing_from_bytes

# ------------ Helpers ------------
def ensure_session():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # each: {"role": "user"/"assistant"/"system", "parts": [{"type":"text","text":...} | {"type":"image","mime":..., "b64":...}]}
    if "parsed_receipts" not in st.session_state:
        st.session_state["parsed_receipts"] = []
    if "parsed_receipts_counter" not in st.session_state:
        st.session_state["parsed_receipts_counter"] = 0


def _safe_json_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "receipt").strip()) or "receipt"
    if not safe.lower().endswith(".json"):
        safe += ".json"
    return safe

def add_message(role: str, text: str = "", images: List[Dict[str, Any]] = None, skip_openai: bool = False):
    msg = {"role": role, "parts": []}
    if skip_openai:
        msg["skip_openai"] = True
    if text:
        msg["parts"].append({"type": "text", "text": text})
    if images:
        for im in images:
            msg["parts"].append({"type": "image", "mime": im["mime"], "b64": im["b64"]})
    st.session_state["messages"].append(msg)

def parts_to_streamlit(msg):
    # Render one chat message to Streamlit bubbles
    with st.chat_message(msg["role"] if msg["role"] in ("user", "assistant") else "assistant"):
        for p in msg["parts"]:
            if p["type"] == "text":
                st.markdown(p["text"])
            elif p["type"] == "image":
                st.image(
                    base64.b64decode(p["b64"]),
                    caption="uploaded image",
                    width=256,
                )

def messages_to_openai_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert internal message structure into OpenAI chat format (supports text + images)."""
    out = []
    for m in messages:
        if m.get("skip_openai"):
            continue
        content = []
        for p in m["parts"]:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{p['mime']};base64,{p['b64']}"}
                })
        # Fallback: if empty content, skip
        if not content:
            continue
        out.append({"role": m["role"], "content": content})
    return out

def encode_upload_to_b64(uploaded_files):
    imgs = []
    raw_entries = []
    if not uploaded_files:
        return imgs, raw_entries
    files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    for uf in files:
        bytes_ = uf.read()
        if not bytes_:
            continue
        b64 = base64.b64encode(bytes_).decode("utf-8")
        mime = uf.type or "image/png"
        imgs.append({"mime": mime, "b64": b64})
        raw_entries.append({
            "bytes": bytes_,
            "name": getattr(uf, "name", None),
            "mime": mime,
        })
    return imgs, raw_entries


# ------------ Streamlit UI ------------
st.set_page_config("Frontier", page_icon="🧾", layout="wide")
st.title("🧾 Receipt Parser")


with st.sidebar:
    system_prompt = st.text_area("System prompt (opsional)", value="You are a helpful assistant.")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Clear chat"):
            st.session_state.clear()
            ensure_session()
            st.rerun()
    with col_b:
        if st.download_button(
            label="Download chat",
            data=json.dumps(st.session_state.get("messages", []), ensure_ascii=False, indent=2),
            file_name="messages.json",
            mime="application/json",
        ):
            pass

ensure_session()
# Inject/refresh system prompt (only once at start or when empty)
if system_prompt and (not st.session_state["messages"] or st.session_state["messages"][0]["role"] != "system"):
    st.session_state["messages"].insert(0, {"role": "system", "parts": [{"type": "text", "text": system_prompt}]})
else:
    # Keep system prompt in sync if user edits it later
    if st.session_state["messages"] and st.session_state["messages"][0]["role"] == "system":
        st.session_state["messages"][0]["parts"] = [{"type": "text", "text": system_prompt}]

chat_placeholder = st.empty()
status_placeholder = st.empty()
receipt_download_placeholder = st.empty()

def render_chat_history():
    with chat_placeholder.container():
        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        for msg in st.session_state["messages"]:
            if msg["role"] == "system":
                continue
            parts_to_streamlit(msg)
        st.markdown("</div>", unsafe_allow_html=True)


def render_parsed_receipts():
    downloads = st.session_state.get("parsed_receipts", [])
    receipt_download_placeholder.empty()
    if not downloads:
        return
    with receipt_download_placeholder.container():
        st.markdown("### Parsed Receipt JSON")
        for idx, entry in enumerate(downloads, start=1):
            file_label = entry.get("name") or f"Receipt {idx}"
            json_payload = entry.get("content", "{}")
            file_name = _safe_json_filename(file_label)
            key_suffix = entry.get("key") or f"download-{file_name}-{idx}"
            st.download_button(
                label=f"Download {file_label}",
                data=json_payload,
                file_name=file_name,
                mime="application/json",
                key=key_suffix,
            )


render_chat_history()
render_parsed_receipts()

# Response helpers
def generate_response_with_spinner(service: OpenAIService, user_text: str):
    try:
        with status_placeholder.container():
            with st.spinner("Generating response..."):
                return service.send_message_with_tokens(user_text)
    finally:
        status_placeholder.empty()


def handle_submit(prompt: str, uploaded_files):
    images_b64, uploaded_raw = encode_upload_to_b64(uploaded_files)

    if not prompt and not images_b64 and not uploaded_raw:
        return

    add_message("user", text=prompt or "", images=images_b64)
    render_chat_history()

    parsed_receipts = []
    if uploaded_raw:
        counter = st.session_state.get("parsed_receipts_counter", 0)
        for idx, raw in enumerate(uploaded_raw, start=1):
            display_name = raw.get("name") or f"Receipt {idx}"
            parse_result = receipt_parsing_from_bytes(raw["bytes"])
            if parse_result and parse_result.content:
                parsed_content = parse_result.content
                try:
                    parsed_json = json.loads(parsed_content)
                    parsed_content = json.dumps(parsed_json, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    parsed_content = parsed_content.strip()
                counter += 1
                parsed_receipts.append({
                    "name": display_name,
                    "content": parsed_content,
                    "key": f"parsed-receipt-{counter}",
                })
                add_message(
                    "assistant",
                    text=f"Parsed receipt ({display_name}):\n```json\n{parsed_content}\n```",
                    skip_openai=True,
                )
            else:
                st.error(f"Failed to parse receipt ({display_name}).")
        if parsed_receipts:
            st.session_state["parsed_receipts"] = parsed_receipts
            st.session_state["parsed_receipts_counter"] = counter
            # Remove uploaded image from the latest user message to avoid re-displaying it
            for msg in reversed(st.session_state["messages"]):
                if msg["role"] == "user":
                    msg["parts"] = [part for part in msg["parts"] if part["type"] != "image"]
                    break
            images_b64 = []
            render_chat_history()
            render_parsed_receipts()

    try:
        openai_service = OpenAIService()
        openai_service.set_system_prompt(system_prompt)

        for msg in st.session_state["messages"]:
            if msg["role"] == "system" or msg.get("skip_openai"):
                continue
            for part in msg["parts"]:
                if part["type"] == "text":
                    openai_service.add_message_to_history(msg["role"], part["text"])

        user_text = prompt or ""
        if images_b64:
            user_text += "\n[User uploaded image(s) attached]"
        if parsed_receipts:
            for entry in parsed_receipts:
                name = entry.get("name", "receipt")
                content = entry.get("content", "")
                user_text += f"\n[Parsed receipt {name}: {content}]"

        result = generate_response_with_spinner(openai_service, user_text)
        add_message("assistant", text=result.content)
        render_chat_history()

    except Exception as e:
        add_message("assistant", text=f"Error: {e}")
        render_chat_history()


# Input area

uploaded = st.file_uploader(
    "Optional: upload image",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False
)
prompt = st.chat_input("Tulis pesan…")
handle_submit(prompt, uploaded)
