import os

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from evaluation.evaluation import run_routing_evaluation, run_rag_evaluation

from graph.workflow import Main_WorkFlow
from agents.clarification import resume_clarification
from services.memory import get_checkpointer
from services.budget import budget_scope
from services.logging_config import configure_logging


def run():
    configure_logging()
    memory , memory_context = get_checkpointer()
    try:
        graph = Main_WorkFlow(memory)

        user_id = os.getenv("DEFAULT_USER_ID", "test-user-1")
        thread_id = os.getenv("DEFAULT_THREAD_ID", "thread-test-user-1")
        permission_level = os.getenv("DEFAULT_PERMISSION_LEVEL", "general")

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "permission_level": permission_level,
            },
        }

        print("Company Intelligence Assistant. Type 'quit' to exit.")
        print(
            f"(user={user_id} | thread={thread_id} | "
            f"permission={permission_level})"
        )

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in {"quit", "exit"}:
                break

            if user_input.lower() == "evaluate":
                print("\nRunning routing evaluation...")
                run_routing_evaluation("routing-eval-v1", graph)
    
                print("\nRunning RAG evaluation...")
                run_rag_evaluation("rag-eval-v1")
    
                print("\nEvaluations complete. Check LangSmith for full results.")
                continue

            with budget_scope():
                result = graph.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )

                while "__interrupt__" in result:
                    interrupt_payload = result["__interrupt__"][0].value
                    question = interrupt_payload["question"]
                    answer = input(f"\n{question}\nYou: ").strip()
                    result = resume_clarification(
                        graph,
                        thread_id,
                        answer,
                        config,
                    )

            messages = result.get("messages") or []
            if messages:
                print(f"\nAssistant: {messages[-1].content}")

            chart_path = result.get("chart_path")
            if chart_path:
                print(f"(chart saved to {chart_path})")
    finally:
        if memory_context is not None:
            memory_context.__exit__(None, None, None)

if __name__ == "__main__":
    run()