from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from state.state import SubGraphSupervisorState
from services.llm import llm
from tools.web_search import web_search, fetch_page

TOOLS = [web_search, fetch_page]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

MAX_ITERATIONS = 5
MAX_SEARCH_ATTEMPTS = 2

RESEARCHER_SYSTEM_PROMPT = """You are a research specialist.
Your only job is to gather high-quality public information about the given task.

Tools:
- web_search(query): find relevant pages
- fetch_page(url): read the main content of a specific page

Strategy:
1. Start with a focused web_search
2. From the results, select the most promising 1-3 URLs and fetch them
3. If the information is still insufficient, you may refine the query and search again (max 2 searches total)
4. When you have enough material, stop calling tools and write a clear intermediate research note that includes:
   - Key findings
   - Sources (title + url)

Rules:
- Never invent information
- Prefer reputable sources
- If after your attempts nothing useful is found, clearly state that.
"""

def Research(state: SubGraphSupervisorState) -> dict:
    if not state.task:
        raise RuntimeError("Research node called with empty task")

    model = llm().bind_tools(TOOLS)

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=f"Research task: {state.task}"),
    ]

    # Continue from previous messages if this is a later research pass
    if state.messages:
        messages = list(state.messages) + [HumanMessage(content="Continue or refine the research based on what you already have.")]

    search_attempts = 0

    for _ in range(MAX_ITERATIONS):
        response = model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # Model finished gathering → return the updated message history
            return {"messages": messages}

        for call in response.tool_calls:
            name = call["name"]
            args = call["args"]
            tool_fn = TOOLS_BY_NAME.get(name)

            if tool_fn is None:
                result = f"Unknown tool: {name}"
            else:
                try:
                    if name == "web_search":
                        search_attempts += 1
                        if search_attempts > MAX_SEARCH_ATTEMPTS:
                            result = "Maximum search attempts reached. Do not search again."
                        else:
                            result = tool_fn.invoke(args)
                    else:
                        result = tool_fn.invoke(args)
                except Exception as e:
                    result = f"Tool error: {type(e).__name__}: {e}"

            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # Hit max iterations
    return {"messages": messages}