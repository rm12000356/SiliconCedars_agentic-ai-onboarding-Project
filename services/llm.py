from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def llm(model: str):
    return ChatGroq(model=model)