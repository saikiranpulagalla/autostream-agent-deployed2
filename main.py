import os
from src.config import GOOGLE_API_KEY  # Load API key from .env
from src.agent_graph import app, update_state_from_user
from src.state import AgentState

DEMO_FLOW = [
    "Hi, tell me about your pricing.",
    "That sounds good, I want to try the Pro plan for my YouTube channel.",
    "John Doe",
    "john@example.com",
    "YouTube"
]

def run_demo():
    """Automated demo matching assignment flow."""
    print("🧪 Running AutoStream Agent Demo...")
    thread = {"configurable": {"thread_id": "demo"}}
    state = AgentState(messages=[], intent="", name="", email="", platform="", lead_captured=False)
    
    for i, user_input in enumerate(DEMO_FLOW, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {user_input}")
        state["messages"].append({"role": "user", "content": user_input})
        state = update_state_from_user(state, user_input)
        
        output = app.invoke(state, thread)
        agent_response = output["messages"][-1]["content"]
        print(f"Agent: {agent_response}")
        state = output
        print(f"State: intent={state['intent']}, name={state.get('name')}, lead_captured={state.get('lead_captured')}")
    
    print("\n✅ Demo Complete: Full flow tested (RAG → Intent → Lead Capture)")

def interactive_mode():
    """Manual terminal chat."""
    thread = {"configurable": {"thread_id": "interactive"}}
    state = AgentState(messages=[], intent="", name="", email="", platform="", lead_captured=False)
    
    print("AutoStream Agent (type 'demo' for auto-run, 'exit' to quit)")
    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() == 'exit':
            break
        if user_input.lower() == 'demo':
            run_demo()
            continue
        
        state["messages"].append({"role": "user", "content": user_input})
        state = update_state_from_user(state, user_input)
        output = app.invoke(state, thread)
        print(f"Agent: {output['messages'][-1]['content']}")
        state = output

if __name__ == "__main__":
    interactive_mode()