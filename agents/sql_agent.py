from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from state.state import SupervisorState, SpecialistResult
from services.llm import llm
from tools.database import GENERAL_TOOLS, ELEVATED_TOOLS
from decimal import Decimal
from groq import BadRequestError


def _extract_chartable_rows(tool_output) -> list[dict] | None:
    if not isinstance(tool_output, list) or not tool_output:
        return None
    if not all(isinstance(row, dict) and len(row) == 2 for row in tool_output):
        return None

    rows = []
    for row in tool_output:
        values = list(row.values())
        label, value = values[0], values[1]
        if isinstance(value, Decimal):
            value = float(value)
        if not isinstance(value, (int, float)):
            return None
        rows.append({"label": str(label), "value": float(value)})
    return rows


def _serialize(obj):
    """Make tool results JSON-serializable for the message history."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def Sql_agent(state: SupervisorState, config: RunnableConfig) -> dict:
    if state.current_task is None:
        raise RuntimeError(
            "SQL node reached with current_task=None. The Supervisor "
            "should always set current_task before routing here."
        )

    configurable = config.get("configurable")
    if not configurable:
        raise RuntimeError("Missing runtime configuration")

    permission_level = configurable.get("permission_level", "general")
    print(f"[SQL] permission_level={permission_level!r}")
    print(f"[SQL] current_task={state.current_task!r}")

    task_lower = state.current_task.lower()
    sensitive = any(k in task_lower for k in ["salary", "salaries", "credential", "password"])

    if permission_level != "elevated" and sensitive:
        return {
            "last_result": SpecialistResult(
                source="sql",
                summary="Salary and credential data require elevated permissions.",
                status="failed",
                issue="permission_denied",
            )
        }

    tools = GENERAL_TOOLS + ELEVATED_TOOLS if permission_level == "elevated" else GENERAL_TOOLS
    tools_by_name = {t.name: t for t in tools}
    print(f"[SQL] bound tools={list(tools_by_name.keys())}")

    model = llm().bind_tools(tools)

    system_message = SystemMessage(
        content=(
            "You are a SQL specialist for an internal company database.\n"
            "Use ONLY the tools provided. Never invent data.\n\n"
            "IMPORTANT RULES:\n"
            "1. Call tools to gather the data you need.\n"
            "2. As soon as you have enough information to answer the user's request, "
            "   STOP calling tools and write a clear final answer in natural language.\n"
            "3. Do NOT call the same tool with the same arguments twice.\n"
            "4. Prefer one efficient query over many small ones when possible.\n\n"
            "Available tables (only these exist):\n"
            "  employees(id, name, department)\n"
            "  sales(id, amount, region, sale_date)\n"
            "  lessons_learned(id, project_name, lesson_text, tags, created_at)\n\n"
            "Elevated-only tools (not queryable via SQL):\n"
            "  get_salary(employee_id) → returns salary\n"
            "  get_user_credential(user_id) → returns credential info\n"
        )
    )

    messages = [system_message, HumanMessage(content=state.current_task)]
    max_iterations = 7
    last_chartable_rows = None
    collected_facts = []          # simple list of what the tools returned

    for i in range(max_iterations):
        print(f"[SQL] iteration {i + 1}/{max_iterations}")
        try:
            response = model.invoke(messages)
        except BadRequestError:
            return {
                "last_result": SpecialistResult(
                    source="sql",
                    summary="This request requires data or tools that are not available for the current permission level.",
                    status="failed",
                    issue="permission_denied",
                )
            }
        messages.append(response)

        if not response.tool_calls:
            # Model produced a final answer
            print(f"[SQL] final answer: {response.content!r}")
            return {
                "last_result": SpecialistResult(
                    source="sql",
                    summary=str(response.content),
                    status="done",
                    structured_data=last_chartable_rows,
                )
            }

        for call in response.tool_calls:
            print(f"[SQL] tool call: {call['name']} args={call['args']}")
            tool_fn = tools_by_name.get(call["name"])

            if tool_fn is None:
                return {
                    "last_result": SpecialistResult(
                        source="sql",
                        summary="This request requires data or tools that are not available for the current permission level.",
                        status="failed",
                        issue="permission_denied",
                    )
                }

            try:
                tool_output = tool_fn.invoke(call["args"])
            except Exception as e:
                err = str(e).lower()
                print(f"[SQL] tool error: {type(e).__name__}: {e}")
                if any(x in err for x in ["does not exist", "undefined table", "undefined column"]):
                    return {
                        "last_result": SpecialistResult(
                            source="sql",
                            summary="The requested data could not be found in the database.",
                            status="failed",
                            issue=f"table_or_schema_missing: {e}",
                        )
                    }
                tool_output = f"Tool execution error: {type(e).__name__}: {e}"

            clean_output = _serialize(tool_output)
            print(f"[SQL] tool result: {str(clean_output)[:300]}")

            # Keep a simple record of what we learned
            collected_facts.append(f"{call['name']}({call['args']}) → {clean_output}")

            detected = _extract_chartable_rows(clean_output)
            if detected:
                last_chartable_rows = detected
                print(f"[SQL] detected chartable rows: {detected}")

            messages.append(
                ToolMessage(content=str(clean_output), tool_call_id=call["id"])
            )

    # ---------------------------------------------------------------
    # Exhausted iterations – but we may already have the data.
    # Synthesize a final answer instead of returning failed.
    # ---------------------------------------------------------------
    if collected_facts:
        print("[SQL] max iterations reached, synthesizing answer from collected facts")
        synthesis_prompt = [
            SystemMessage(content=(
                "You are a helpful assistant. Below are the exact results returned by database tools. "
                "Write a clear, natural-language answer to the user's original request using ONLY this data. "
                "Do not invent anything."
            )),
            HumanMessage(content=(
                f"Original request: {state.current_task}\n\n"
                f"Tool results:\n" + "\n".join(collected_facts)
            )),
        ]
        try:
            final = llm().invoke(synthesis_prompt)
            summary = str(final.content)
            status = "done"
            issue = None
        except Exception as e:
            summary = "Could not complete the SQL request within the allowed steps."
            status = "failed"
            issue = f"tool-calling loop exceeded max_iterations + synthesis failed: {e}"
    else:
        summary = "Could not complete the SQL request within the allowed steps."
        status = "failed"
        issue = "tool-calling loop exceeded max_iterations"

    return {
        "last_result": SpecialistResult(
            source="sql",
            summary=summary,
            status=status,
            structured_data=last_chartable_rows,
            issue=issue,
        )
    }