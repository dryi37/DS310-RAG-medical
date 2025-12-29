import streamlit as st
import requests
from chat_store import ChatStore

API_URL = "http://127.0.0.1:8000/chat"
store = ChatStore("chat_ui.db")

st.set_page_config(page_title="Medical RAG Chatbot", layout="wide")

# ---------- init session ----------
if "conversation_id" not in st.session_state:
    cid = store.new_conversation("New chat")
    st.session_state.conversation_id = cid

if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

# lưu debug của câu trả lời gần nhất (không ghi vào SQLite)
if "last_debug" not in st.session_state:
    st.session_state.last_debug = None

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## 💬 Chats")

    if st.button("➕ New chat", use_container_width=True):
        cid = store.new_conversation("New chat")
        st.session_state.conversation_id = cid
        st.session_state.last_debug = None
        st.rerun()

    chats = store.list_conversations(limit=30)
    for c in chats:
        label = c["title"]
        if st.button(label, key=f"chat_{c['id']}", use_container_width=True):
            st.session_state.conversation_id = c["id"]
            st.session_state.last_debug = None  # đổi chat thì reset debug
            st.rerun()

    st.divider()
    st.session_state.show_debug = st.checkbox(
        "Show debug info",
        value=st.session_state.show_debug
    )

# ---------- main ----------
st.title("🩺 Medical RAG Chatbot (Demo)")

cid = st.session_state.conversation_id
messages = store.get_messages(cid)

# render history
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# nếu bật debug, hiển thị panel cho response gần nhất
if st.session_state.show_debug and st.session_state.last_debug:
    with st.expander("🔎 Debug (last response)", expanded=True):
        st.json(st.session_state.last_debug)

# chat input
user_input = st.chat_input("Nhập câu hỏi...")

if user_input:
    store.add_message(cid, "user", user_input)

    payload = {"conversation_id": cid, "query": user_input}

    try:
        r = requests.post(API_URL, json=payload, timeout=180)

        if r.status_code != 200:
            # show lỗi dưới dạng message
            store.add_message(cid, "assistant", f"[ERROR] Backend error {r.status_code}: {r.text}")
            st.session_state.last_debug = {
                "status_code": r.status_code,
                "raw_text": r.text[:2000],
            }
            st.rerun()

        data = r.json()
        answer = data.get("answer", "")

        # auto set title (first user message in this conversation)
        if len(messages) == 0:
            title = user_input.strip()[:40]
            store.rename_conversation(cid, title)

        store.add_message(cid, "assistant", answer)

        # lưu debug (không lưu vào history)
        st.session_state.last_debug = {
            "route": data.get("route"),
            "rewritten_query": data.get("rewritten_query"),
            "latency_ms": data.get("latency_ms"),
            "top_docs_count": len(data.get("top_docs", []) or []),
            "top_docs": data.get("top_docs", []) if st.session_state.show_debug else [],
        }

    except Exception as e:
        store.add_message(cid, "assistant", f"[ERROR] Request failed: {e}")
        st.session_state.last_debug = {"exception": str(e)}

    st.rerun()
