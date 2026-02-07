import os
import sys

# Ensure local imports work
sys.path.append(os.getcwd())

from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, List
from summarisation import summarization_graph, SummaryState

# === Define Parent Graph State ===
# The parent graph state must be compatible with or able to map to the subgraph state.
# Here we'll just extend SummaryState or use a compatible structure.
class ParentState(TypedDict):
    user_request: str
    summary_result: str
    # Fields required for mapping to SummaryState if we want direct passing
    input_type: str
    content: str
    extracted_text: str
    chunks: List[str]
    summaries: List[str]
    final_summary: str
    metadata: Dict[str, Any]
    error: str

# === Define a Node that calls the Subgraph ===
# In LangGraph, we can add a compiled graph as a node directly if the state schema matches.
# Or we can wrap it in a function if we need to transform state.
# Let's try adding it directly as a node, assuming state sharing.

def setup_node(state: ParentState) -> ParentState:
    print("Parent: Setting up summarization task...")
    # Pre-populate state for the subgraph
    return {
        "input_type": "text",
        "content": state["user_request"],
        "extracted_text": "",
        "chunks": [],
        "summaries": [],
        "final_summary": "",
        "metadata": {},
        "error": ""
    }

def handle_result_node(state: ParentState) -> ParentState:
    print("Parent: Handling result...")
    return {"summary_result": state.get("final_summary", "No summary generated")}

# === Build Parent Graph ===
workflow = StateGraph(ParentState)

workflow.add_node("setup", setup_node)
workflow.add_node("summarizer", summarization_graph) # Adding the subgraph as a node
workflow.add_node("result_handler", handle_result_node)

workflow.set_entry_point("setup")
workflow.add_edge("setup", "summarizer")
workflow.add_edge("summarizer", "result_handler")
workflow.add_edge("result_handler", END)

app = workflow.compile()

# === Run Integration Test ===
print("Starting Integration Test...")
initial_state = {
    "user_request": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.",
    # Initialize other fields to avoid KeyError if TypedDict is strict (though typically it's loose in runtime unless validated)
    "input_type": "", "content": "", "extracted_text": "", "chunks": [], "summaries": [], "final_summary": "", "metadata": {}, "error": ""
}

try:
    result = app.invoke(initial_state)
    print("\n=== INTEGRATION SUCCESSFUL ===")
    print(f"Final Output in Parent: {result['summary_result']}")
except Exception as e:
    print(f"\n=== INTEGRATION FAILED ===")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
