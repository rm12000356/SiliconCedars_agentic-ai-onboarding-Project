from typing import Literal, get_args
from state.state import SupervisorState, SubGraphSupervisorState

MainRoute = Literal[
    "rag",
    "convo",
    "sql",
    "research",
    "visu",
    "clarification",
    "end",
]

SubRoute = Literal[
    "researcher",
    "report",
    "end",
]

_VALID_MAIN_ROUTES = set(get_args(MainRoute))
_VALID_SUB_ROUTES = set(get_args(SubRoute))


def route_supervisor(state: SupervisorState) -> MainRoute:
    """
    Route the main graph based on the Supervisor's decision.
    """
    next_route = state.next

    if next_route is None:
        raise RuntimeError(
            "Routing error: state.next is None. The Supervisor node either "
            "hasn't run yet or didn't set 'next' in its return value. Check "
            "graph edge ordering and the Supervisor node's return dict."
        )

    if next_route not in _VALID_MAIN_ROUTES:
        raise RuntimeError(
            f"State corruption: Supervisor state.next={next_route!r} is not a valid "
            f"MainRoute. This should be unreachable if SupervisorDecision validation "
            f"upstream is working correctly."
        )

    return next_route


def route_sub_supervisor(state: SubGraphSupervisorState) -> SubRoute:
    """
    Route the research subgraph based on the Sub-Supervisor's decision.
    """
    next_route = state.next

    if next_route is None:
        raise RuntimeError(
            "Routing error: sub-supervisor state.next is None. The sub-supervisor "
            "node either hasn't run yet or didn't set 'next' in its return value."
        )
    
    if next_route not in _VALID_SUB_ROUTES:
        raise RuntimeError(
            f"State corruption: sub-supervisor state.next={next_route!r} is not a "
            f"valid SubRoute. This should be unreachable if SubRouteDecision "
            f"validation upstream is working correctly."
        )

    return next_route