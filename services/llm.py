from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Supported LLMs
#"openai/gpt-oss-120b",
#"openai/gpt-oss-20b",
#"qwen/qwen3.6-27b",
#"openai/gpt-oss-safeguard-20b",

# this project only works on openai/gpt-oss-120b the others free eather got a very low rate limit or are not reasoning at all sending wrong outputs.

def llm(model: str = "openai/gpt-oss-120b"):
    return ChatGroq(model=model)