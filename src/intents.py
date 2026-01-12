from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Lazy initialization: llm will be created only when detect_intent() is called
llm = None
_current_api_key = None

INTENT_PROMPT = ChatPromptTemplate.from_template(
    """Classify the user's intent based on the message: {message}
    Possible intents:
    - casual_greeting: Simple hi/hello or non-product related.
    - product_inquiry: Questions about pricing, features, policies.
    - high_intent: User expresses readiness to sign up, try, or buy (e.g., 'I want to try Pro').
    Respond ONLY with the intent name (e.g., 'product_inquiry')."""
)

def detect_intent(message: str, api_key: str = None) -> str:
    """Detects user intent using LLM prompt.
    
    Args:
        message: The user's message to classify
        api_key: Optional API key. If provided, uses this; otherwise tries environment
    """
    global llm, _current_api_key
    
    # Use provided api_key, fallback to environment
    key_to_use = api_key or os.environ.get("GOOGLE_API_KEY")
    
    # Recreate llm if API key changed (for per-session isolation)
    if llm is None or _current_api_key != key_to_use:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, api_key=key_to_use)
        _current_api_key = key_to_use
    
    chain = INTENT_PROMPT | llm
    response = chain.invoke({"message": message})
    return response.content.strip().lower()