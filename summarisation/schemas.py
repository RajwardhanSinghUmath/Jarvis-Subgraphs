from typing import TypedDict, List, Dict, Any

class SummaryState(TypedDict):
    """State management for the summarization graph."""
    input_type: str  # text, pdf, url, email, video, audio, digest
    content: str  # Raw content or path
    extracted_text: str  # Extracted text from various sources
    chunks: List[str]  # Text chunks for processing
    summaries: List[str]  # Individual chunk summaries
    final_summary: str  # Final consolidated summary
    metadata: Dict[str, Any]  # Additional metadata
    error: str  # Error messages if any
