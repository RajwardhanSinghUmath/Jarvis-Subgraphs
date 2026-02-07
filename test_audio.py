import os
import sys

# Add the current directory to sys.path to ensure imports work correctly
sys.path.append(os.getcwd())

from summarisation.graph import summarization_graph

# Use absolute path for robustness
audio_file = "scene_7.mp3"
audio_path = os.path.abspath(audio_file)
print(f"Testing audio summarization with file: {audio_path}")

if not os.path.exists(audio_path):
    print(f"Error: Audio file '{audio_path}' not found!")
    sys.exit(1)

# Initialize state
state = {
    "input_type": "audio",
    "content": audio_path,
    "extracted_text": "",
    "chunks": [],
    "summaries": [],
    "final_summary": "",
    "metadata": {},
    "error": ""
}

try:
    print("Invoking summarization graph...")
    # Invoke the graph
    result = summarization_graph.invoke(state)
    
    # Check for errors in the result state
    if result.get("error"):
        print(f"Error in graph execution: {result['error']}")
    else:
        print("\n=== FINAL SUMMARY ===\n")
        print(result["final_summary"])
        print("\n=====================\n")
        print(f"Metadata: {result.get('metadata')}")

except Exception as e:
    print(f"Exception occurred: {e}")
