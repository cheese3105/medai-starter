from langgraph.graph import StateGraph, END

from state import AgentState
from agents.reasoning_agent import make_reasoning_agent_node
from run_config import RunConfig


def build_graph(run_config: RunConfig):
    """v0: chỉ có Reasoning Agent. v1+: thêm Retrieval Agent phía trước khi
    `run_config.retrieval.enabled = true` (bật/tắt qua YAML, không sửa code).

    Node/edge được thêm có điều kiện - variant nào không bật retrieval
    (v0.yaml) chạy graph giống hệt như trước, không có gì thay đổi.

    Verifier Agent (bước sau) sẽ nối theo đúng pattern này khi
    run_config.verifier.enabled.
    """
    graph = StateGraph(AgentState)

    graph.add_node("reasoning_agent", make_reasoning_agent_node(run_config))

    if run_config.retrieval.enabled:
        from agents.retrieval_agent import make_retrieval_agent_node

        graph.add_node("retrieval_agent", make_retrieval_agent_node(run_config))
        graph.set_entry_point("retrieval_agent")
        graph.add_edge("retrieval_agent", "reasoning_agent")
    else:
        graph.set_entry_point("reasoning_agent")

    graph.add_edge("reasoning_agent", END)

    return graph.compile()
