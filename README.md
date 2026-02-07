# Summarization Subgraph

A reusable LangGraph subgraph for multi-modal summarization.

## Structure
*   **`summarisation/`**: The core package.
    *   `graph.py`: Contains the `summarization_graph`.
    *   `nodes.py`: Implementation of summarization nodes.
    *   `schemas.py`: Typed state definition.
*   **`requirements.txt`**: Project dependencies.

## Usage

```python
from summarisation import summarization_graph

state = {
    "input_type": "text",  # text, pdf, url, email, video, audio, digest
    "content": "Your content here...",
    "extracted_text": "",
    "chunks": [],
    "summaries": [],
    "final_summary": "",
    "metadata": {},
    "error": ""
}

result = summarization_graph.invoke(state)

if result.get("error"):
    print(f"Error: {result['error']}")
else:
    print(result["final_summary"])
```

## Features
*   **Seven Input Types**: Text, PDF, URL, Email, Video (YouTube), Audio, Smart Digest.
*   **Intelligent Chunking**: Splits long text recursively.
*   **Two-Stage Summarization**: Summarizes chunks then consolidates.
*   **Robust Error Handling**: Errors are captured in state, not crashed.
# Jarvis-Subgraphs
# Jarvis-Subgraphs
# Jarvis-Subgraphs
# Jarvis-Subgraphs
