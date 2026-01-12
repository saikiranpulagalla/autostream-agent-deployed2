import streamlit as st
from typing import Dict, Any

def chat_message(role: str, content: str):
    """Renders chat bubble with role-aware styling."""
    if role == "user":
        st.chat_message("user", avatar="👤")
    else:
        st.chat_message("assistant", avatar="🤖")
    st.markdown(content)

def lead_form(state: Dict[str, Any]) -> Dict[str, Any]:
    """Progressive lead collection form based on state."""
    if state.get("collecting_lead"):
        if not state.get("name"):
            name = st.text_input("What's your name?", key="name_input")
            if st.button("Submit Name") and name:
                state["name"] = name
                st.rerun()
        elif not state.get("email"):
            email = st.text_input("Your email?", key="email_input")
            if st.button("Submit Email") and email:
                state["email"] = email
                st.rerun()
        elif not state.get("platform"):
            platform = st.selectbox("Creator platform?", ["YouTube", "Instagram", "TikTok", "Other"], key="platform_input")
            if st.button("Submit Platform") and platform:
                state["platform"] = platform
                st.rerun()
    return state

def display_state_info(state: Dict[str, Any]):
    """Sidebar: Debug state (intent, collected data, lead status)."""
    with st.sidebar:
        st.header("Agent State")
        st.write(f"**Intent:** {state.get('intent', 'None')}")
        st.write(f"**Name:** {state.get('name', 'Pending')}")
        st.write(f"**Email:** {state.get('email', 'Pending')}")
        st.write(f"**Platform:** {state.get('platform', 'Pending')}")
        st.write(f"**Lead Captured:** {'✅ Yes' if state.get('lead_captured') else '❌ No'}")