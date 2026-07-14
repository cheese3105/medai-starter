from langgraph.graph import StateGraph, END

from state import AgentState
from agents.reasoning_agent import make_reasoning_agent_node
from run_config import RunConfig


def build_graph(run_config: RunConfig):
    """Bước 1: chỉ có Reasoning Agent - nhận run_config để build node
    (model/prompt_version/temperature/seed đã "khoá" theo variant).

    Ở các bước sau, thêm Retrieval/Verifier Agent: chỉ cần
    graph.add_node(...) + graph.add_edge(...) hoặc add_conditional_edges(...)
    tuỳ run_config.retrieval.enabled / run_config.verifier.enabled.
    """
    graph = StateGraph(AgentState)

    graph.add_node("reasoning_agent", make_reasoning_agent_node(run_config))

    graph.set_entry_point("reasoning_agent")
    graph.add_edge("reasoning_agent", END)

    return graph.compile()
