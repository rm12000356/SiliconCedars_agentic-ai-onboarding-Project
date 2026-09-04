import uuid
from langchain_core.messages import HumanMessage

from langchain_core.runnables import RunnableConfig
from graph.workflow import Main_WorkFlow
from agents.clarification import resume_clarification
from state.state import SupervisorState


def run():
    graph = Main_WorkFlow()

    # No auth/identity layer wired up yet, so thread_id is just a fresh
    # random session per process run. Ties to the user_id/thread_id
    # design in the handoff doc once auth actually exists.
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    print("Company Intelligence Assistant. Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        result = graph.invoke(
            SupervisorState(messages=[HumanMessage(content=user_input)]),
            config=config,
        )

        # A single turn can pause more than once if the Supervisor
        # routes to clarification, gets an answer, and is still unsure.
        while "__interrupt__" in result:
            interrupt_payload = result["__interrupt__"][0].value
            question = interrupt_payload["question"]
            answer = input(f"\n{question}\nYou: ").strip()
            result = resume_clarification(graph, thread_id, answer)

        last_message = result["messages"][-1]
        print(f"\nAssistant: {last_message.content}")


if __name__ == "__main__":
    run()