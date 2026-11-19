import os
import streamlit as st
from langgraph.types import Command
from agent import DEFAULT_THREAD_ID, IDEAS_STORE, TASKS_STORE, build_graph

st.set_page_config(page_title="Content Creator Agent", page_icon="🎬", layout="wide")
st.title("🎬 Content Creator AI Assistant")

with st.sidebar:
    st.header("სესია")
    groq_key = st.text_input(
        "GROQ_API_KEY",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
    )
    thread_id = st.text_input("Thread ID", value=DEFAULT_THREAD_ID)

    if st.button("ახალი სესია"):
        TASKS_STORE[thread_id] = []
        IDEAS_STORE[thread_id] = []
        st.session_state.pop("graph", None)
        st.session_state.pop("pending_interrupt", None)
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.subheader("დავალებები")
    for t in TASKS_STORE.get(thread_id, []):
        st.write(f"#{t['id']} — {t['title']}")

    st.subheader("კონტენტ-იდეები")
    for i in IDEAS_STORE.get(thread_id, []):
        st.write(f"#{i['id']} [{i['category']}] {i['idea']}")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph(api_key=groq_key or None)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

config = {"configurable": {"thread_id": thread_id}}
graph = st.session_state.graph

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

pending = st.session_state.pending_interrupt
if pending:
    st.warning(pending.get("message", "დაადასტურეთ წაშლა."))
    col1, col2 = st.columns(2)
    if col1.button("✅ ვადასტურებ"):
        res = graph.invoke(Command(resume=True), config)
        st.session_state.pending_interrupt = None
        st.session_state.chat_history.append(("assistant", res["messages"][-1].content))
        st.rerun()
    if col2.button("❌ უარს ვამბობ"):
        res = graph.invoke(Command(resume=False), config)
        st.session_state.pending_interrupt = None
        st.session_state.chat_history.append(("assistant", res["messages"][-1].content))
        st.rerun()

prompt = st.chat_input("დაწერე შეტყობინება...")
if prompt:
    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    res = graph.invoke({"messages": [("user", prompt)]}, config)
    interrupts = res.get("__interrupt__")
    if interrupts:
        payload = interrupts[0].value
        st.session_state.pending_interrupt = payload
        st.session_state.chat_history.append(("assistant", payload.get("message", "დაადასტურეთ მოქმედება.")))
        st.rerun()
    else:
        reply = res["messages"][-1].content
        st.session_state.chat_history.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)
