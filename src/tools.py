from langchain_core.tools import tool

@tool
def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock API to capture lead. Only call when all details are collected."""
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    return f"Lead captured: {name}, {email}, {platform}"