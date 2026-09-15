from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from state.state import SubGraphSupervisorState
from services.llm import llm
from tools.web_search import web_search, fetch_page

TOOLS = [web_search, fetch_page]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

MAX_ITERATIONS = 6
MAX_SEARCH_ATTEMPTS = 1
MAX_FETCH_ATTEMPTS = 3

RESEARCHER_SYSTEM_PROMPT = """You are a research specialist.
Your only job is to gather high-quality public information about the given task.

Tools:
- web_search(query): find relevant pages (returns title, url, snippet)
- fetch_page(url, snippet): read a page. Always pass snippet from the
  matching search result. If the page is blocked, the snippet is used.

Strategy:
1. web_search once
2. fetch 1-3 of the most relevant URLs (pass each hit's snippet)
3. If a fetch returns "[Page fetch failed...]", treat that snippet as
   weak evidence. Do not invent. You may still write the note from
   snippets if no full page loaded — mark those sources as snippets.
4. When you have enough material, stop calling tools and write the research note
   (key findings + sources).

Rules:
- Never invent information
- Prefer reputable sources
- If after your attempts nothing useful is found, clearly state that.
"""

WRITE_NOTE_NOW = (
    "No more tool calls. Write the research note NOW from the search "
    "snippets and any page text you already have. Include key findings "
    "and sources (title + url). Mark snippet-only sources as snippets. "
    "Do not invent numbers that were not in those texts."
)


def _final_note(messages: list) -> dict:
    """Force one untool'd invoke so gathered snippets aren't thrown away."""
    messages.append(HumanMessage(content=WRITE_NOTE_NOW))
    try:
        response = llm().invoke(messages)
    except Exception as e:
        print(f"[RESEARCH] final note failed: {type(e).__name__}: {e}")
        return {
            "research_messages": [
                AIMessage(
                    content=(
                        "Research ended without a clean note. See tool "
                        "results above for snippets and any fetched text."
                    )
                )
            ]
        }

    # Safety: strip any accidental tool calls from the final answer
    if isinstance(response, AIMessage) and response.tool_calls:
        response.tool_calls = []

    return {"research_messages": [response]}


def Research(state: SubGraphSupervisorState) -> dict:
    if not state.task:
        raise RuntimeError("Research node called with empty task")

    model = llm().bind_tools(TOOLS)

    # Collect any previous research notes already in the subgraph
    if not state.research_messages:
        prior_notes = ""
    else:
        prior_notes = "\n\n".join(
            str(m.content) for m in state.research_messages if isinstance(m, AIMessage)
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
    fetch_attempts = 0

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

        # Model decided to stop → return its final answer
        if not response.tool_calls:
            return {"research_messages": [response]}

        # Execute tool calls
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
                            result = (
                                "Maximum search attempts reached. "
                                "Do not search again. Fetch remaining URLs "
                                "from the first search or write the note."
                            )
                            print("[RESEARCH] search attempt limit hit")
                        else:
                            result = tool_fn.invoke(args)

                    elif name == "fetch_page":
                        fetch_attempts += 1
                        if fetch_attempts > MAX_FETCH_ATTEMPTS:
                            result = (
                                "Maximum page fetches reached. "
                                "Write the research note now from what you already have."
                            )
                            print("[RESEARCH] fetch attempt limit hit")
                        else:
                            result = tool_fn.invoke(args)

                    else:
                        result = tool_fn.invoke(args)

                except Exception as e:
                    result = f"Tool error: {type(e).__name__}: {e}"
                    print(f"[RESEARCH] tool error: {type(e).__name__}: {e}")

            print(f"[RESEARCH] tool result: {str(result)[:300]!r}")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        # Soft early-exit: we already have a search + enough pages
        if search_attempts >= MAX_SEARCH_ATTEMPTS and fetch_attempts >= 2:
            print("[RESEARCH] enough material gathered → forcing final note")
            return _final_note(messages)

    # Hard stop
    print("[RESEARCH] max iterations reached → forcing final note")
    return _final_note(messages)