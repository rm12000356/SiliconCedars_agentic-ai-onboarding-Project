from graph.workflow import main_workflow, sub_workflow


def test_main_graph_compiles():
    graph = main_workflow()
    nodes = set(graph.get_graph().nodes)
    for required in (
        "memory_manager", "supervisor", "sql", "rag", "research",
        "visu", "convo", "clarification", "finalize",
    ):
        assert required in nodes, f"missing node {required}: {nodes}"


def test_research_subgraph_compiles():
    graph = sub_workflow()
    nodes = set(graph.get_graph().nodes)
    for required in ("controller", "research", "report"):
        assert required in nodes