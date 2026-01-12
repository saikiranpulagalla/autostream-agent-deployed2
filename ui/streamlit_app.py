import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env FIRST - before any src imports
load_dotenv()

# Add parent directory to path FIRST - before any src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# Initialize session state for API key BEFORE importing src modules
# IMPORTANT: DO NOT load API key from .env for deployed app (security + privacy)
# Each user must provide their own API key per session
# The key is cleared on page refresh automatically
if "openai_api_key" not in st.session_state:
    # Check if we're in local development (has .env file with actual key)
    # Only use .env key if it's a placeholder (development only)
    env_key = os.getenv("GOOGLE_API_KEY", "")
    if env_key and not env_key.startswith("your_"):
        # Local development - pre-fill from .env
        st.session_state.openai_api_key = env_key
    else:
        # Deployed/Production - require user input
        st.session_state.openai_api_key = None

# IMPORTANT: Do NOT set os.environ from user input - it's global and shared across sessions!
# Instead, we'll pass the API key explicitly to each function

from src.agent_graph import app, update_state_from_user  # Core LangGraph
from src.state import AgentState
from src.intents import detect_intent

st.set_page_config(page_title="AutoStream Agent Demo", layout="wide")

# ============ HELPER FUNCTION TO PROCESS MESSAGE ============
def process_message(user_msg: str, is_test_mode: bool = False):
    """Process a user message through the agent pipeline."""
    st.session_state.agent_trace = []
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    st.session_state.agent_state["messages"].append({"role": "user", "content": user_msg})
    
    # Update trace
    st.session_state.agent_trace.append("🔄 Processing user input...")
    st.session_state.agent_trace.append(f"📝 Message: {user_msg}")
    
    # Update state for lead collection (before intent detection)
    st.session_state.agent_state = update_state_from_user(st.session_state.agent_state, user_msg)
    
    # Detect intent and add to trace
    try:
        # Pass API key explicitly to detect_intent for per-session isolation
        intent = detect_intent(user_msg, api_key=st.session_state.openai_api_key)
        st.session_state.agent_state["intent"] = intent
        st.session_state.agent_trace.append(f"🎯 Intent: **{intent}**")
        
        if intent == "product_inquiry":
            st.session_state.agent_trace.append("📚 Retrieving from knowledge base...")
        elif intent == "high_intent":
            st.session_state.agent_trace.append("⭐ High-intent lead detected!")
        else:
            st.session_state.agent_trace.append("💬 Casual greeting")
    except Exception as e:
        st.session_state.agent_trace.append(f"⚠️ Intent error: {str(e)}")
    
    # Run LangGraph (persistent thread)
    st.session_state.agent_trace.append("🤖 Agent processing...")
    try:
        # Temporarily set environment variable for this invocation only
        # This ensures the agent_node gets the current session's API key
        original_key = os.environ.get("GOOGLE_API_KEY")
        if st.session_state.openai_api_key:
            os.environ["GOOGLE_API_KEY"] = st.session_state.openai_api_key
        
        output = app.invoke(
            st.session_state.agent_state,
            {"configurable": {"thread_id": st.session_state.thread_id}}
        )
        
        # Restore original environment (remove session-specific key)
        if original_key:
            os.environ["GOOGLE_API_KEY"] = original_key
        elif "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
        
        st.session_state.agent_trace.append("✅ Response generated")
        
        # Extract & display response
        last_msg = output["messages"][-1]
        if isinstance(last_msg, dict):
            agent_response = last_msg["content"]
        else:
            agent_response = last_msg.content
        
        st.session_state.chat_history.append({"role": "assistant", "content": agent_response})
        st.session_state.agent_state = output  # Sync state
        st.session_state.agent_trace.append("📤 Response sent")
        
        if output.get("lead_captured"):
            st.session_state.agent_trace.append("🎉 Lead captured!")
    except Exception as e:
        st.session_state.agent_trace.append(f"❌ Error: {str(e)}")
        st.error(f"Agent Error: {str(e)}")

# ============ SIDEBAR: API KEY MANAGEMENT ============
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Determine if we're in local dev or deployed
    env_key = os.getenv("GOOGLE_API_KEY", "")
    is_local_dev = env_key and not env_key.startswith("your_")
    
    if is_local_dev:
        st.info(
            "✅ **Local Development Mode**\n\n"
            "API key loaded from `.env` file\n\n"
            "To test on Streamlit Cloud, paste your own key below."
        )
    else:
        st.info(
            "🔒 **Streamlit Cloud / Public Deployment**\n\n"
            "Each session requires its own API key (never stored permanently)"
        )
    
    # Check if API key is already set in session state
    if not st.session_state.openai_api_key:
        st.warning("⚠️ API key required to use this app")
        api_key_input = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")
        if api_key_input:
            # Validate basic format
            if len(api_key_input) < 20:
                st.error("❌ Invalid API key format (too short)")
            else:
                # Store ONLY in session state - DO NOT set os.environ (it's global!)
                st.session_state.openai_api_key = api_key_input
                st.success("✅ API key loaded for this session only")
                st.rerun()  # Rerun to re-evaluate the sidebar and show success state
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("✅ API key active")
        with col2:
            if st.button("🔄 Change", key="change_key_btn"):
                st.session_state.openai_api_key = None
                st.rerun()
        
        with st.expander("📋 API Key Info"):
            masked_key = "AIzaSy***" + st.session_state.openai_api_key[-4:]
            st.code(masked_key, language="text")
            if is_local_dev:
                st.caption("📁 Loaded from `.env` (Local Development)")
            else:
                st.caption("🔒 Session-only storage (cleared on refresh)")

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "demo_thread"
if "agent_state" not in st.session_state:
    st.session_state.agent_state = AgentState(
        messages=[], intent="", name="", email="", platform="", lead_captured=False
    )
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_trace" not in st.session_state:
    st.session_state.agent_trace = []
if "waiting_for_lead_info" not in st.session_state:
    st.session_state.waiting_for_lead_info = False

# ============ MAIN LAYOUT ============
st.title("🚀 AutoStream Social-to-Lead Agent")
st.markdown("**Test all flows:** Pricing Q&A → High-Intent → Lead Capture")

# Create main columns
col_chat, col_trace = st.columns([3, 1])

# ============ CHAT INTERFACE (Main Column) ============
with col_chat:
    st.markdown("### 💬 Conversation")
    
    # Sidebar: Quick Test Buttons
    st.sidebar.markdown("### 🎯 Quick Test Buttons")
    st.sidebar.caption("💡 Send test messages to the ongoing conversation")
    col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
    
    with col_btn1:
        if st.button("👋 Test Greeting", use_container_width=True):
            st.session_state.agent_trace = ["🔄 Processing test: Casual Greeting"]
            st.session_state.waiting_for_lead_info = False
            user_msg = "Hi, how are you?"
            process_message(user_msg, is_test_mode=True)
            st.rerun()
    
    with col_btn2:
        if st.button("💬 Test Pricing", use_container_width=True):
            st.session_state.agent_trace = ["🔄 Processing test: Pricing Inquiry"]
            st.session_state.waiting_for_lead_info = False
            user_msg = "Hi, tell me about your pricing."
            process_message(user_msg, is_test_mode=True)
            st.rerun()
    
    with col_btn3:
        if st.button("🎥 Test High Intent", use_container_width=True):
            st.session_state.agent_trace = ["🔄 Processing test: High Intent"]
            st.session_state.waiting_for_lead_info = False
            user_msg = "That sounds good, I want to try the Pro plan for my YouTube channel."
            process_message(user_msg, is_test_mode=True)
            st.rerun()
    
    if st.sidebar.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.agent_state = AgentState(
            messages=[], intent="", name="", email="", platform="", lead_captured=False
        )
        st.session_state.agent_trace = []
        st.session_state.waiting_for_lead_info = False
        st.rerun()
    
    # Display chat history with improved styling
    st.markdown("#### 💬 Chat History")
    chat_container = st.container(height=450, border=True)
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                # User message - right aligned with blue background
                st.markdown(
                    f"""
                    <div style="
                        background-color: #E3F2FD;
                        border-left: 4px solid #1976D2;
                        padding: 12px 15px;
                        border-radius: 8px;
                        margin-bottom: 10px;
                        margin-left: 50px;
                        color: #000000;
                    ">
                        <b style="color: #1976D2;">👤 You:</b><br><span style="color: #333333;">{msg['content']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # Agent message - left aligned with green background
                st.markdown(
                    f"""
                    <div style="
                        background-color: #E8F5E9;
                        border-left: 4px solid #388E3C;
                        padding: 12px 15px;
                        border-radius: 8px;
                        margin-bottom: 10px;
                        margin-right: 50px;
                        color: #000000;
                    ">
                        <b style="color: #388E3C;">🤖 Agent:</b><br><span style="color: #333333;">{msg['content']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # ============ PROGRESSIVE LEAD FORM ============
    # Only show if high-intent and not all fields collected
    if st.session_state.agent_state.get("intent") == "high_intent" and not st.session_state.agent_state.get("lead_captured"):
        st.markdown("### 📋 Lead Information (Required to Proceed)")
        st.info("💡 Please provide all three details below to complete your registration.")
        
        # Get current values
        name = st.session_state.agent_state.get("name", "")
        email = st.session_state.agent_state.get("email", "")
        platform = st.session_state.agent_state.get("platform", "")
        
        # Show all three fields together
        col1, col2, col3 = st.columns(3)
        
        with col1:
            name_input = st.text_input(
                "Your Name:",
                value=name,
                placeholder="e.g., John Doe",
                key="name_field"
            )
        
        with col2:
            email_input = st.text_input(
                "Your Email:",
                value=email,
                placeholder="e.g., john@example.com",
                key="email_field"
            )
        
        with col3:
            platform_input = st.text_input(
                "Your Platform:",
                value=platform,
                placeholder="e.g., YouTube, Instagram",
                key="platform_field"
            )
        
        st.markdown("---")
        
        # Check if all three fields are filled
        if name_input and email_input and platform_input:
            # All fields are filled, show submit button
            st.success("✅ All required information provided!")
            if st.button("📤 Submit Lead Information", use_container_width=True, type="primary"):
                # Update state with all three values
                st.session_state.agent_state["name"] = name_input
                st.session_state.agent_state["email"] = email_input
                st.session_state.agent_state["platform"] = platform_input
                
                # Add a special message to trigger the agent to process the lead
                combined_msg = f"My details are: Name: {name_input}, Email: {email_input}, Platform: {platform_input}"
                with st.spinner("⏳ Processing lead..."):
                    process_message(combined_msg)
                st.rerun()
        else:
            # Show which fields are missing
            st.warning("⏳ Please fill in all three fields above to proceed")
            if name_input:
                st.caption("✅ Name filled")
            if email_input:
                st.caption("✅ Email filled")
            if platform_input:
                st.caption("✅ Platform filled")
        
        # Block regular chat input while collecting lead info
        st.session_state.waiting_for_lead_info = True
    else:
        st.session_state.waiting_for_lead_info = False
    
    # ============ CHAT INPUT ============
    st.markdown("---")
    if st.session_state.agent_state.get("lead_captured"):
        st.success("✅ Lead captured successfully! Feel free to ask any other questions.")
    if st.session_state.waiting_for_lead_info:
        st.warning("📋 Please complete the lead information form above before continuing the conversation.")
    elif not st.session_state.waiting_for_lead_info:
        if prompt := st.chat_input("Type your message..."):
            with st.spinner("⏳ Agent thinking..."):
                process_message(prompt)
            st.rerun()

# ============ AGENT TRACE (Side Column) ============
with col_trace:
    st.markdown("### 🔍 Agent Trace")
    trace_container = st.container(height=400, border=True)
    with trace_container:
        if st.session_state.agent_trace:
            for trace in st.session_state.agent_trace:
                st.markdown(f"• {trace}")
        else:
            st.markdown("*No activity yet.*\n*Send a message to see agent trace.*")
    
    st.markdown("### 📊 State Info")
    state_box = st.container(border=True, height=300)
    with state_box:
        st.markdown(f"**Intent:** `{st.session_state.agent_state.get('intent', 'None')}`")
        st.markdown(f"**Name:** `{st.session_state.agent_state.get('name', '❌ Not set')}`")
        st.markdown(f"**Email:** `{st.session_state.agent_state.get('email', '❌ Not set')}`")
        st.markdown(f"**Platform:** `{st.session_state.agent_state.get('platform', '❌ Not set')}`")
        if st.session_state.agent_state.get('lead_captured'):
            st.markdown(f"**Lead Captured:** 🟢 **Yes**")
        else:
            st.markdown(f"**Lead Captured:** 🔴 No")
        
        st.markdown("---")
        msg_count = len(st.session_state.agent_state.get("messages", []))
        st.markdown(f"**Messages:** {msg_count}")

# Success Banner
if st.session_state.agent_state.get("lead_captured"):
    st.success("🎉 **Lead Captured Successfully!** Check console/terminal for mock API output.")
