import os
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from graph.workflow import Main_WorkFlow
from agents.clarification import resume_clarification


def run():
    graph = Main_WorkFlow()

    # Stable identities – same values across restarts so the checkpointer
    # and long-term memory keep working.
    user_id = os.getenv("DEFAULT_USER_ID", "test-user-1")
    thread_id = os.getenv("DEFAULT_THREAD_ID", "thread-test-user-1")
    permission_level = os.getenv("DEFAULT_PERMISSION_LEVEL", "elevated")

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "permission_level": permission_level,
        },
    }

    print("Company Intelligence Assistant. Type 'quit' to exit.")
    print(f"(user={user_id} | thread={thread_id} | permission={permission_level})")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        while "__interrupt__" in result:
            interrupt_payload = result["__interrupt__"][0].value
            question = interrupt_payload["question"]
            answer = input(f"\n{question}\nYou: ").strip()
            result = resume_clarification(graph, thread_id, answer, config)

        last_message = result["messages"][-1]
        print(f"\nAssistant: {last_message.content}")


if __name__ == "__main__":
    run()