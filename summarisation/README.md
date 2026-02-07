# Summarization Subgraph

A reusable LangGraph subgraph for multi-modal summarization.

## Features
*   **Seven Input Types**: Text, PDF, URL, Email, Video (YouTube), Audio, Smart Digest.
*   **Intelligent Chunking**: Recursively splits long text into manageable chunks.
*   **Two-Stage Summarization**: Summarizes chunks independently, then consolidates them into a final summary.
*   **State Management**: Uses `SummaryState` for typed state management.

## Usage

```python
from summarisation import summarization_graph

state = {
    "input_type": "text",
    "content": "Your long text here...",
    "extracted_text": "",
    "chunks": [],
    "summaries": [],
    "final_summary": "",
    "metadata": {},
    "error": ""
}

result = summarization_graph.invoke(state)
print(result["final_summary"])
```

## Input Types
*   `text`: Direct text string.
*   `pdf`: Path to PDF file.
*   `url`: Web page URL.
*   `email`: Email content string.
*   `video`: YouTube video URL (requires `yt-dlp`).
*   `audio`: Audio file path (requires `whisper`).
*   `digest`: JSON string of a list of text items.

## Error Handling
The graph logic includes `try/except` blocks. If an error occurs, it is captured in the `error` field of the state. It does not raise an exception, allowing the parent graph to handle the failure gracefully.
