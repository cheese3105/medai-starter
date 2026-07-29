"""LangGraph for Chat mode (v3).

Flow: Retrieval Agent → Chat Reasoning Agent → Chat Verifier Agent

Reuses Retrieval Agent from benchmark (it only returns evidence, format-agnostic).
Uses chat-specific Reasoning and Verifier agents for free-form output.
"""

from langgraph.graph import StateGraph, END

from state import AgentState
from agents.retrieval_agent import make_retrieval_agent_node
from agents.chat_reasoning_agent import make_chat_reasoning_node
from agents.chat_verifier_agent import make_chat_verifier_node
from run_config import RunConfig


def build_chat_graph(run_config: RunConfig):
    """Build graph for chat mode.

    Uses AgentState as state schema (required by LangGraph to pass keys between nodes).
    Chat mode passes choices=[] since there are no multiple-choice options.
    """
    graph = StateGraph(AgentState)

    graph.add_node("reasoning_agent", make_chat_reasoning_node(run_config))

    if run_config.retrieval.enabled:
        graph.add_node("retrieval_agent", make_retrieval_agent_node(run_config))
        graph.set_entry_point("retrieval_agent")
        graph.add_edge("retrieval_agent", "reasoning_agent")
    else:
        graph.set_entry_point("reasoning_agent")

    if run_config.verifier.enabled:
        graph.add_node("verifier_agent", make_chat_verifier_node(run_config))
        graph.add_edge("reasoning_agent", "verifier_agent")
        graph.add_edge("verifier_agent", END)
    else:
        graph.add_edge("reasoning_agent", END)

    return graph.compile()
