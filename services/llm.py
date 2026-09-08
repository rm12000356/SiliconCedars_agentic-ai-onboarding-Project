from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import SecretStr
import os

load_dotenv()

PRIMARY_MODEL = "openai/gpt-oss-120b"

FALLBACK_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",                 
    "openai/gpt-oss-safeguard-20b",
]

OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def llm(model: str | None = None):

    # 1. Caller forced a specific model
    if model is not None:
        if model.startswith("openrouter/") or model.startswith("openai/") and "openrouter" in model:
            # allow forcing OpenRouter
            return _make_openrouter(model.replace("openrouter/", ""))
        return ChatGroq(model=model)

    # 2 + 3. Try Groq models
    candidates = [PRIMARY_MODEL] + FALLBACK_MODELS
    for m in candidates:
        try:
            print(f"[LLM] Trying Groq: {m}")
            client = ChatGroq(model=m)
            client.invoke("ping")
            return client
        except Exception as e:
            print(f"[LLM] Groq {m} failed: {e}")
            continue

    # 4. OpenRouter last resort
    try:
        print(f"[LLM] All Groq models failed → OpenRouter ({OPENROUTER_MODEL})")
        return _make_openrouter(OPENROUTER_MODEL)
    except Exception as e:
        print(f"[LLM] OpenRouter also failed: {e}")

    raise RuntimeError(
        "All configured LLM models failed. "
        "Check GROQ_API_KEY, OPENROUTER_API_KEY and model names."
    )


def _make_openrouter(model_name: str) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing from .env")

    return ChatOpenAI(
        model=model_name,
        api_key=SecretStr(api_key),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/rm12000356/SiliconCedars_agentic-ai-onboarding-Project",
            "X-Title": "SiliconCedars",
        },
    )