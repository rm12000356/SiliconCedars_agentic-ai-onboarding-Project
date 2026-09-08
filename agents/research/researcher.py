from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from state.state import SubGraphSupervisorState
from services.llm import llm
from tools.web_search import web_search, fetch_page

TOOLS = [web_search, fetch_page]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

MAX_ITERATIONS = 5
MAX_SEARCH_ATTEMPTS = 1

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
    """
    Runs a bounded tool-calling loop to gather information. The loop's
    internal scaffolding (system prompt, individual tool calls, raw
    tool outputs like full fetched page text) stays entirely local to
    this function call, it is never written to shared subgraph state.
    Only the final condensed research note is persisted, since that's
    the only part later steps (another research pass, the report
    writer) actually need. This keeps state.messages small regardless
    of how many tool calls happened internally to produce it.
    """
    if not state.task:
        raise RuntimeError("Research node called with empty task")

    model = llm().bind_tools(TOOLS)

    # Local scratchpad: previous condensed notes (if any) provide context
    # for continuing, but the loop itself starts fresh each pass, it does
    # not replay old raw tool transcripts.
    if not state.messages:
        prior_notes = ""
    else:
        prior_notes = "\n\n".join(
            str(m.content) for m in state.messages if isinstance(m, AIMessage)
        )

    task_prompt = f"Research task: {state.task}"
    if prior_notes:
        task_prompt += (
            f"\n\nPrevious research notes so far:\n{prior_notes}\n\n"
            "Continue or refine the research based on what's already been found. "
            "Do not repeat what's already covered, focus on filling gaps."
        )

    print(f"[RESEARCH] task={state.task!r}")
    print(f"[RESEARCH] prior_notes={prior_notes[:300]!r}")

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=task_prompt),
    ]

    search_attempts = 0

    for _ in range(MAX_ITERATIONS):
        try:
            response = model.invoke(messages)
        except Exception as e:
        
            print(f"[RESEARCH] model call failed: {type(e).__name__}: {e}")
            messages.append(
                HumanMessage(
                    content=(
                        f"Your previous response could not be processed: {e}. "
                        "Please try again, making sure any tool call arguments "
                        "are valid JSON."
                    )
                )
            )
            continue

        messages.append(response)

        if not response.tool_calls:

            return {"messages": [response]}

        for call in response.tool_calls:
            name = call["name"]
            args = call["args"]
            tool_fn = TOOLS_BY_NAME.get(name)

            print(f"[RESEARCH] tool call: {name} args={args}")
            
            if tool_fn is None:
                result = f"Unknown tool: {name}"
                print(f"[RESEARCH] unknown tool requested: {name}")
            else:
                try:
                    if name == "web_search":
                        search_attempts += 1
                        if search_attempts > MAX_SEARCH_ATTEMPTS:
                            result = "Maximum search attempts reached. Do not search again."
                            print("[RESEARCH] search attempt limit hit")
                        else:
                            result = tool_fn.invoke(args)
                    else:
                        result = tool_fn.invoke(args)
                except Exception as e:
                    result = f"Tool error: {type(e).__name__}: {e}"
                    print(f"[RESEARCH] tool error: {type(e).__name__}: {e}")

            print(f"[RESEARCH] tool result: {str(result)[:300]!r}")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    print("[RESEARCH] max iterations reached without a final note")
    return {
        "messages": [
            AIMessage(content="Research did not reach a conclusion within the allowed steps.")
        ]
    }