from datetime import datetime
import os
import json
import re
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from .schemas import SummaryState
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
# Note: Ideally this should be passed in or configured, but for this refactor we keep it here
llm = ChatGroq(
    api_key=os.getenv("GROQ_API"),
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# ============= EXTRACTION NODES =============

def extract_text_node(state: SummaryState) -> SummaryState:
    """Extract text from direct text input"""
    try:
        state["extracted_text"] = state["content"]
        state["metadata"] = {"source": "direct_text", "length": len(state["content"])}
    except Exception as e:
        state["error"] = f"Text extraction error: {str(e)}"
    return state


def extract_pdf_node(state: SummaryState) -> SummaryState:
    """Extract text from PDF file"""
    try:
        loader = PyPDFLoader(state["content"])
        pages = loader.load()
        extracted = "\n\n".join([page.page_content for page in pages])
        state["extracted_text"] = extracted
        state["metadata"] = {
            "source": "pdf",
            "pages": len(pages),
            "file_path": state["content"]
        }
    except Exception as e:
        state["error"] = f"PDF extraction error: {str(e)}"
    return state


def extract_url_node(state: SummaryState) -> SummaryState:
    """Extract text from URL"""
    try:
        loader = WebBaseLoader(state["content"])
        docs = loader.load()
        extracted = "\n\n".join([doc.page_content for doc in docs])
        state["extracted_text"] = extracted
        state["metadata"] = {
            "source": "url",
            "url": state["content"]
        }
    except Exception as e:
        state["error"] = f"URL extraction error: {str(e)}"
    return state


def extract_email_node(state: SummaryState) -> SummaryState:
    """Extract text from email (expects email content as string)"""
    try:
        # Assuming email is passed as formatted string
        state["extracted_text"] = state["content"]
        state["metadata"] = {
            "source": "email",
            "length": len(state["content"])
        }
    except Exception as e:
        state["error"] = f"Email extraction error: {str(e)}"
    return state


def extract_video_node(state: SummaryState) -> SummaryState:
    """Extract transcript from video URL (YouTube)"""
    try:
        import yt_dlp
        
        # Extract video ID from URL (handle both youtube.com and youtu.be formats)
        url = state["content"]
        if "youtu.be/" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0]
        elif "v=" in url:
            video_id = url.split("v=")[-1].split("&")[0]
        else:
            raise ValueError("Invalid YouTube URL format")
        
        # Configure yt-dlp to get subtitles
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Try to get subtitles
            subtitles = info.get('subtitles', {})
            automatic_captions = info.get('automatic_captions', {})
            
            transcript_text = ""
            
            # Try manual subtitles first
            if 'en' in subtitles:
                # Get the subtitle data
                for sub in subtitles['en']:
                    if 'url' in sub:
                        import requests
                        response = requests.get(sub['url'])
                        # Parse VTT or SRT format
                        transcript_text = response.text
                        break
            # Fall back to automatic captions
            elif 'en' in automatic_captions:
                for sub in automatic_captions['en']:
                    if 'url' in sub:
                        import requests
                        response = requests.get(sub['url'])
                        transcript_text = response.text
                        break
            
            # Clean up the transcript (remove VTT/SRT formatting)
            if transcript_text:
                lines = transcript_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    # Skip timestamp lines and empty lines
                    if '-->' not in line and line.strip() and not line.strip().isdigit():
                        # Remove VTT tags
                        line = line.replace('<c>', '').replace('</c>', '')
                        if not line.startswith('WEBVTT') and not line.startswith('Kind:'):
                            cleaned_lines.append(line.strip())
                
                extracted = ' '.join(cleaned_lines)
            else:
                # If no subtitles available, use video description as fallback
                extracted = info.get('description', '')
        
        if not extracted:
            raise ValueError("No transcript or description available for this video")
        
        state["extracted_text"] = extracted
        state["metadata"] = {
            "source": "video",
            "video_id": video_id,
            "url": state["content"],
            "title": info.get('title', 'Unknown')
        }
    except Exception as e:
        state["error"] = f"Video extraction error: {str(e)}. Make sure yt-dlp is installed (pip install yt-dlp)."
    return state


def extract_audio_node(state: SummaryState) -> SummaryState:
    """Extract text from audio file using Whisper"""
    try:
        import whisper
        
        model = whisper.load_model("base")
        result = model.transcribe(state["content"])
        
        state["extracted_text"] = result["text"]
        state["metadata"] = {
            "source": "audio",
            "file_path": state["content"],
            "language": result.get("language", "unknown")
        }
    except Exception as e:
        state["error"] = f"Audio extraction error: {str(e)}. Make sure whisper is installed."
    return state


def extract_digest_node(state: SummaryState) -> SummaryState:
    """Handle multiple content items for smart digest"""
    try:
        # Content should be a JSON string of list of items
        items = json.loads(state["content"])
        
        combined_text = ""
        for idx, item in enumerate(items):
            combined_text += f"\n\n=== Item {idx + 1} ===\n{item}\n"
        
        state["extracted_text"] = combined_text
        state["metadata"] = {
            "source": "digest",
            "num_items": len(items)
        }
    except Exception as e:
        state["error"] = f"Digest extraction error: {str(e)}"
    return state


# ============= PROCESSING NODES =============

def chunk_text_node(state: SummaryState) -> SummaryState:
    """Split text into manageable chunks"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_text(state["extracted_text"])
        state["chunks"] = chunks
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["num_chunks"] = len(chunks)
    except Exception as e:
        state["error"] = f"Chunking error: {str(e)}"
    return state


def summarize_chunks_node(state: SummaryState) -> SummaryState:
    """Summarize individual chunks"""
    try:
        summaries = []
        
        system_prompt = """You are an expert summarizer. Create a concise, accurate summary 
        of the provided text. Focus on key points, main ideas, and important details. 
        Keep the summary clear and well-structured."""
        
        for idx, chunk in enumerate(state["chunks"], 1):
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Summarize this text:\n\n{chunk}")
            ]
            
            response = llm.invoke(messages)
            summaries.append(response.content)
        
        state["summaries"] = summaries
    except Exception as e:
        state["error"] = f"Chunk summarization error: {str(e)}"
    return state


def final_summary_node(state: SummaryState) -> SummaryState:
    """Create final consolidated summary"""
    try:
        if len(state["summaries"]) == 1:
            state["final_summary"] = state["summaries"][0]
        else:
            combined_summaries = "\n\n".join(state["summaries"])
            
            system_prompt = """You are an expert at creating comprehensive summaries. 
            Combine the following partial summaries into one cohesive, well-structured final summary. 
            Ensure all key points are covered without redundancy."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Create a final summary from these partial summaries:\n\n{combined_summaries}")
            ]
            
            response = llm.invoke(messages)
            state["final_summary"] = response.content
    except Exception as e:
        state["error"] = f"Final summarization error: {str(e)}"
    return state
