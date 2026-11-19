
import streamlit as st
import uuid

from agent import build_graph
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

# ----------------------------------
# 1. გვერდის კონფიგურაცია
# ----------------------------------

st.set_page_config(
    page_title="Trendora AI",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Trendora AI")
st.caption("Your Personal Content Manager")

# ----------------------------------
# 2. Session State
# ----------------------------------

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------
# 3. Thread configuration
# ----------------------------------

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

# ----------------------------------
# 4. Chat history display
# ----------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------------
# 5. Approval Flow
# ----------------------------------

def process_result(result):

    interrupts = result.get("__interrupt__", [])

    if interrupts:
        interrupt_data = interrupts[0].value

        st.session_state.pending_approval = interrupt_data

        st.warning(
            "⚠️ დადასტურება საჭიროა:\n\n"
            + str(interrupt_data)
        )

    else:
        st.session_state.pending_approval = None

        messages = result.get("messages", [])

        if messages:
            last_message = messages[-1]

            if hasattr(last_message, "content"):
                answer = str(last_message.content)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                st.session_state.history.append({
                    "role": "assistant",
                    "content": answer
                })

# ----------------------------------
# 6. Approval buttons
# ----------------------------------

if st.session_state.pending_approval is not None:

    st.warning("გსურთ მოქმედების დადასტურება?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ დადასტურება", use_container_width=True):

            result = st.session_state.graph.invoke(
                Command(resume=True),
                config=config
            )

            process_result(result)
            st.rerun()

    with col2:
        if st.button("❌ უარყოფა", use_container_width=True):

            result = st.session_state.graph.invoke(
                Command(resume=False),
                config=config
            )

            process_result(result)
            st.rerun()

# ----------------------------------
# 7. Chat input
# ----------------------------------

if st.session_state.pending_approval is None:

    prompt = st.chat_input("რა კონტენტის შექმნაში დაგეხმარო?")

    if prompt:

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        st.session_state.history.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Trendora AI მუშაობს..."):

                result = st.session_state.graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config
                )

                process_result(result)

        st.rerun()
