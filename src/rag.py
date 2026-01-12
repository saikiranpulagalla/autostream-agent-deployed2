import json
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Lazy initialization: llm will be created only when rag_retrieve() is called
llm = None
_current_api_key = None
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load knowledge base
with open("data/knowledge_base.json", "r") as f:
    kb_data = json.load(f)

# Convert to documents for RAG
documents = []
for section, content in kb_data.items():
    documents.append(Document(page_content=json.dumps(content), metadata={"section": section}))

# Create vector store
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an AutoStream assistant. Answer the question using ONLY this context: {context}
    
Conversation history for context:
{conversation_context}

Question: {question}

If the information is not in the context, say 'I don't have that information in my knowledge base.'
If the user refers to something mentioned before in the conversation, acknowledge and use that context."""
)

def rag_retrieve(question: str, conversation_context: str = "", api_key: str = None) -> str:
    """Retrieves and generates answer from knowledge base using RAG with conversation context.
    
    Args:
        question: The user's question
        conversation_context: Previous conversation for context
        api_key: Optional API key. If provided, uses this; otherwise tries environment
    """
    global llm, _current_api_key
    
    # Use provided api_key, fallback to environment
    key_to_use = api_key or os.environ.get("GOOGLE_API_KEY")
    
    # Recreate llm if API key changed (for per-session isolation)
    if llm is None or _current_api_key != key_to_use:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, api_key=key_to_use)
        _current_api_key = key_to_use
    
    docs = retriever.invoke(question)
    if not docs:
        return "I don't have that information in my knowledge base."
    context = docs[0].page_content
    chain = RAG_PROMPT | llm
    response = chain.invoke({
        "context": context, 
        "question": question,
        "conversation_context": conversation_context
    })
    return response.content