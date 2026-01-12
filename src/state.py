from typing import TypedDict, Annotated, List
from langgraph.graph import add_messages

class AgentState(TypedDict):
    """State schema for the agent graph. Manages conversation history and collected lead data."""
    messages: Annotated[List[dict], add_messages]  # Conversation history (user/agent messages)
    intent: str  # Current detected intent (e.g., 'casual_greeting', 'product_inquiry', 'high_intent')
    name: str  # Collected name (for high-intent lead)
    email: str  # Collected email
    platform: str  # Collected platform (e.g., YouTube)
    lead_captured: bool  # Flag to indicate if lead capture tool has been called