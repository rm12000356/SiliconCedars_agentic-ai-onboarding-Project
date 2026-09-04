from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state.state import SupervisorState, SubGraphSupervisorState

from graph.routing import route_supervisor, route_sub_supervisor

from agents.conversation_agent import Convo
from agents.rag_agent import RAG
from agents.research.report_writer import Report_W
from agents.research.researcher import Research
from agents.research.supervisor import Sub_controler
from agents.supervisor import supervisor_agent
from agents.sql_agent import Sql_agent
from agents.visualization_agent import Visualization
from agents.clarification import Clarification
from agents.research.research_node import make_research_node


def Main_WorkFlow():

    subgraph = sub_workflow()


    builder = StateGraph(SupervisorState)

    
    builder.add_node("supervisor", supervisor_agent)
    builder.add_node("rag", RAG)
    builder.add_node("convo", Convo)
    builder.add_node("sql", Sql_agent)
    builder.add_node("visu", Visualization)
    builder.add_node("research", make_research_node(subgraph))
    builder.add_node("clarification", Clarification)

    
    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
        {
            "rag": "rag",
            "convo": "convo",
            "sql": "sql",
            "visu": "visu",
            "research": "research",
            "clarification": "clarification",
            "end": END,
        },
    )

    
    builder.add_edge("rag", "supervisor")
    builder.add_edge("convo", "supervisor")
    builder.add_edge("sql", "supervisor")
    builder.add_edge("visu", "supervisor")
    builder.add_edge("research", "supervisor")
    builder.add_edge("clarification", "supervisor")


    memory = MemorySaver()

    graph =builder.compile(checkpointer=memory)

    png = graph.get_graph().draw_mermaid_png()

    with open("graph_structure.png", "wb") as f:
        f.write(png)
        
    return graph

def sub_workflow():

    builder = StateGraph(SubGraphSupervisorState)

    builder.add_node("controler", Sub_controler)
    builder.add_node("research", Research)
    builder.add_node("report", Report_W)

    builder.add_edge(START, "controler")

    builder.add_conditional_edges(
        "controler",
        route_sub_supervisor,
        {
            "researcher": "research",
            "report": "report",
            "end": END,
        }
    )

    builder.add_edge("research", "controler")
    builder.add_edge("report", "controler")

    graph =builder.compile()
    
    png = graph.get_graph().draw_mermaid_png()

    with open("SubGraph_structure.png", "wb") as f:
        f.write(png)
        
    return graph

if __name__ == "__main__":
    Main_WorkFlow()