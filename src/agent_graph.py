from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
import os
from src.state import AgentState
from src.intents import detect_intent
from src.rag import rag_retrieve
from src.tools import mock_lead_capture

# Lazy initialization: llm will be created only when agent_node() is called
llm = None
_current_api_key = None
tools = [mock_lead_capture]
llm_with_tools = None

def agent_node(state: AgentState, api_key: str = None) -> AgentState:
    """Main agent node: Detects intent, responds based on conversation history.
    
    Args:
        state: The current agent state
        api_key: Optional API key for LLM calls
    """
    global llm, _current_api_key
    
    # Use provided api_key, fallback to environment
    key_to_use = api_key or os.environ.get("GOOGLE_API_KEY")
    
    # Initialize llm only on first call or when key changes (lazy initialization with per-session isolation)
    if llm is None or _current_api_key != key_to_use:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, api_key=key_to_use)
        _current_api_key = key_to_use
    
    # Get last message
    last_msg = state["messages"][-1]
    if isinstance(last_msg, dict):
        last_message = last_msg["content"]
    else:
        last_message = last_msg.content
    
    # Check if lead is captured and user asks to edit/update details
    if state.get("lead_captured"):
        edit_keywords = ["edit", "update", "change", "modify", "fix", "wrong", "correct", "mistake"]
        if any(keyword.lower() in last_message.lower() for keyword in edit_keywords):
            response = "Thanks for letting me know. For security reasons, updates to submitted details are handled by our team. Please reply to the confirmation email, or our team will assist you shortly."
            state["messages"] = add_messages(state["messages"], [{"role": "assistant", "content": response}])
            return state
    
    intent = detect_intent(last_message, api_key=key_to_use)
    state["intent"] = intent

    if intent == "product_inquiry":
        # Use RAG for knowledge-based response with conversation context
        # Build context from last few messages
        context_msgs = []
        for msg in state["messages"][-5:]:  # Last 5 messages for context
            if isinstance(msg, dict):
                context_msgs.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
            else:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                context_msgs.append(f"{role}: {msg.content}")
        
        conversation_context = "\n".join(context_msgs)
        response = rag_retrieve(last_message, conversation_context, api_key=key_to_use)
    elif intent == "high_intent":
        # If already captured, don't ask again
        if state.get("lead_captured"):
            response = "Thanks! Your details have been captured successfully. Our team will reach out to you shortly. Feel free to ask if you have other questions about AutoStream."
        # Check if all details are collected at once via form submission
        elif "My details are:" in last_message and state.get("name") and state.get("email") and state.get("platform"):
            response = "Perfect! 🎉 I've received all your information. Let me process your registration now..."
        # Otherwise, check if we have all fields from previous steps
        elif state.get("name") and state.get("email") and state.get("platform"):
            response = "Excellent! 🎉 I have all your information. Processing your lead registration now..."
        # If we don't have all fields, ask for them
        else:
            response = "Great! To get started, please share your name, email, and creator platform."
    else:
        # Casual or default - use conversation context
        context_msgs = []
        for msg in state["messages"][-7:]:  # Last 7 messages for better context
            if isinstance(msg, dict):
                context_msgs.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
            else:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                context_msgs.append(f"{role}: {msg.content}")
        
        conversation_context = "\n".join(context_msgs)
        
        # Use LLM with conversation context for natural responses
        from langchain_core.prompts import ChatPromptTemplate
        casual_prompt = ChatPromptTemplate.from_template(
            """You are AutoStream's AI assistant — a helpful, intelligent, and disciplined conversational agent.

AutoStream is a SaaS product that provides automated video editing tools for content creators, especially YouTube creators.

Your goal is not just to chat, but to follow a correct agentic workflow: understand user intent, answer accurately, and capture leads only when appropriate.

Rules:
1. Always read the full conversation history before responding.
2. Classify each user message into exactly one intent:
   - casual_greeting
   - product_or_pricing_query
   - high_intent_lead
3. Handle intents as follows:
   - casual_greeting: respond briefly and friendly, do not explain product details, ask how you can help.
   - product_or_pricing_query: answer using the AutoStream knowledge base, be clear and accurate, do not ask for name, email, or platform.
   - high_intent_lead: switch to lead qualification mode and collect details one by one: name, email, creator platform.
4. Tool usage rule: call the lead capture tool only after name, email, and platform are all available. Never call it prematurely.
5. Memory: remember previously shared details, do not re-ask, and allow intent to shift naturally.
6. Style: be natural, friendly, professional, concise, and avoid hallucinating features or pricing.
7. If intent is unclear, ask a short clarifying question instead of assuming.

Conversation history:
{context}

Respond to the user's latest message by strictly following the rules above."""
        )
        
        chain = casual_prompt | llm
        response = chain.invoke({"context": conversation_context}).content

    state["messages"] = add_messages(state["messages"], [{"role": "assistant", "content": response}])
    return state

def tool_condition(state: AgentState) -> str:
    """Decides if tool should be called."""
    if state["intent"] == "high_intent" and state.get("name") and state.get("email") and state.get("platform") and not state.get("lead_captured"):
        return "tool_node"
    return END

def tool_node(state: AgentState) -> AgentState:
    """Executes tools if needed."""
    import uuid
    from langchain_core.messages import ToolMessage
    
    tool_call = mock_lead_capture.invoke({
        "name": state["name"],
        "email": state["email"],
        "platform": state["platform"]
    })
    state["lead_captured"] = True
    
    # Create a proper ToolMessage with tool_call_id
    tool_message = ToolMessage(
        content=tool_call,
        tool_call_id=str(uuid.uuid4()),
        name="mock_lead_capture"
    )
    state["messages"] = add_messages(state["messages"], [tool_message])
    return state

def update_state_from_user(state: AgentState, user_message: str) -> AgentState:
    """Updates state based on user input for lead collection."""
    if state["intent"] == "high_intent":
        if not state.get("name"):
            state["name"] = user_message.strip()
        elif not state.get("email"):
            state["email"] = user_message.strip()
        elif not state.get("platform"):
            state["platform"] = user_message.strip()
    return state

# Build the graph
graph = StateGraph(state_schema=AgentState)
graph.add_node("agent_node", agent_node)
graph.add_node("tool_node", tool_node)
graph.add_conditional_edges("agent_node", tool_condition, {"tool_node": "tool_node", END: END})
graph.add_edge("tool_node", END)
graph.add_edge(START, "agent_node")

# Compile with memory (checkpointer for state persistence across turns)
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)