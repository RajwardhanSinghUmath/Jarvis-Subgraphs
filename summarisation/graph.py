from langgraph.graph import StateGraph, END
from .schemas import SummaryState
from .nodes import (
    extract_text_node,
    extract_pdf_node,
    extract_url_node,
    extract_email_node,
    extract_video_node,
    extract_audio_node,
    extract_digest_node,
    chunk_text_node,
    summarize_chunks_node,
    final_summary_node
)

def route_by_input_type(state: SummaryState) -> str:
    """Route to appropriate extraction node based on input type"""
    if state.get("error"):
        return "error"
    
    input_type = state["input_type"]
    routing = {
        "text": "extract_text",
        "pdf": "extract_pdf",
        "url": "extract_url",
        "email": "extract_email",
        "video": "extract_video",
        "audio": "extract_audio",
        "digest": "extract_digest"
    }
    
    return routing.get(input_type, "error")

def create_summarization_graph():
    """Create the LangGraph workflow"""
    workflow = StateGraph(SummaryState)
    
    # Add extraction nodes
    workflow.add_node("extract_text", extract_text_node)
    workflow.add_node("extract_pdf", extract_pdf_node)
    workflow.add_node("extract_url", extract_url_node)
    workflow.add_node("extract_email", extract_email_node)
    workflow.add_node("extract_video", extract_video_node)
    workflow.add_node("extract_audio", extract_audio_node)
    workflow.add_node("extract_digest", extract_digest_node)
    
    # Add processing nodes
    workflow.add_node("chunk_text", chunk_text_node)
    workflow.add_node("summarize_chunks", summarize_chunks_node)
    workflow.add_node("final_summary", final_summary_node)
    
    # Set entry point with conditional routing
    workflow.set_conditional_entry_point(
        route_by_input_type,
        {
            "extract_text": "extract_text",
            "extract_pdf": "extract_pdf",
            "extract_url": "extract_url",
            "extract_email": "extract_email",
            "extract_video": "extract_video",
            "extract_audio": "extract_audio",
            "extract_digest": "extract_digest",
            "error": END
        }
    )
    
    # Connect extraction nodes to chunking
    for node in ["extract_text", "extract_pdf", "extract_url", "extract_email", 
                 "extract_video", "extract_audio", "extract_digest"]:
        workflow.add_edge(node, "chunk_text")
    
    # Connect processing pipeline
    workflow.add_edge("chunk_text", "summarize_chunks")
    workflow.add_edge("summarize_chunks", "final_summary")
    workflow.add_edge("final_summary", END)
    
    return workflow.compile()

# Publicly exposed graph
summarization_graph = create_summarization_graph()
