from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from state.state import SupervisorState, SpecialistResult
from services.llm import llm
from tools.database import GENERAL_TOOLS, ELEVATED_TOOLS
 
 
def Sql_agent(state: SupervisorState, config: RunnableConfig) -> dict:
    """
    Runs a bounded tool-calling loop against the permission-gated tool
    set. The model can call tools, see results, and call more tools,
    until it produces a final answer with no more tool calls, or the
    iteration limit is hit.
 
    Permission gating happens here, in plain Python, before the model
    ever runs: if permission_level isn't "elevated", the elevated
    tools are simply never in the list the model is bound to. The
    model cannot call a tool it was never given.
    """
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
 
    if permission_level == "elevated":
        tools = GENERAL_TOOLS + ELEVATED_TOOLS
    else:
        tools = GENERAL_TOOLS
 
    tools_by_name = {t.name: t for t in tools}
 
    print(f"[SQL] bound tools={list(tools_by_name.keys())}")
 
    model = llm().bind_tools(tools)
 
    system_message = SystemMessage(
        content=(
            "You are a SQL specialist for an internal company database. "
            "Use only the tools provided to you. Do not invent data. "
            "When you have enough information, give a clear final answer "
            "in natural language and stop calling tools.\n\n"
            "The following tables and columns are the ONLY ones that exist. "
            "Never guess or assume a column name that isn't listed here, "
            "if you need something not listed, say so instead of guessing.\n\n"
            "employees(id, name, department)\n"
            "sales(id, amount, region, sale_date)\n"
            "lessons_learned(id, project_name, lesson_text, tags, created_at)\n\n"
            "The following tables only exist for elevated-permission tools, "
            "not for run_general_query, and are not directly queryable:\n"
            "salaries(employee_id, salary) -- use the get_salary tool, not raw SQL\n"
            "credentials(user_id, password_hash) -- use the get_user_credential tool, not raw SQL"
        )
    )
 
    messages = [system_message] + [HumanMessage(content=state.current_task)]
    max_iterations = 5
 
    for i in range(max_iterations):
        print(f"[SQL] iteration {i + 1}/{max_iterations}")
        response = model.invoke(messages)
        messages = messages + [response]
 
        if not response.tool_calls:
            # Model gave a final answer, no more tools requested.
            print(f"[SQL] final answer: {response.content!r}")
            result = SpecialistResult(
                source="sql",
                summary= str(response.content),
                status="done",
            )
            return {"last_result": result}
 
        for call in response.tool_calls:
            print(f"[SQL] tool call: {call['name']} args={call['args']}")
            tool_fn = tools_by_name.get(call["name"])
 
            if tool_fn is None:
                print(f"[SQL] blocked unavailable tool: {call['name']}")
                result = SpecialistResult(
                    source="sql",
                    summary=(
                        "This request requires data or tools that are not "
                        "available for the current permission level."
                    ),
                    status="failed",
                    issue="permission_denied",
                )
                return {"last_result": result}
 
            try:
                tool_output = tool_fn.invoke(call["args"])
            except Exception as e:
                err = str(e).lower()
                print(f"[SQL] tool error: {type(e).__name__}: {e}")
 
                # Missing table / schema style errors → fail clearly for Supervisor
                if (
                    "does not exist" in err
                    or "undefined table" in err
                    or "undefined column" in err
                    or "relation" in err
                    and "does not exist" in err
                ):
                    result = SpecialistResult(
                        source="sql",
                        summary="The requested data could not be found in the database.",
                        status="failed",
                        issue=f"table_or_schema_missing: {e}",
                    )
                    return {"last_result": result}
 
                # Other DB/tool errors → feed back to the model so it can retry
                tool_output = f"Tool execution error: {type(e).__name__}: {e}"
 
            print(f"[SQL] tool result: {str(tool_output)[:300]}")
            messages = messages + [
                ToolMessage(content=str(tool_output), tool_call_id=call["id"])
            ]
 
    # Exhausted max_iterations without a final answer.
    result = SpecialistResult(
        source="sql",
        summary="Could not complete the SQL request within the allowed steps.",
        status="failed",
        issue="tool-calling loop exceeded max_iterations",
    )
    return {"last_result": result}