import ollama
import asyncio
import time
import random
import uuid
import logging
import requests
import os
import re
import threading
import hashlib
import traceback
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import uvicorn
from datetime import datetime
from tts_engine import set_voice_style, speak
set_voice_style(r"C:\Users\Admin\Documents\AI\voice.wav")
from tts_engine import (
    TTS_QUEUE,
    tts_engine,
    speak,
    set_voice_style,
    toggle_mute,
    tts_consumer
)

tts_queue = TTS_QUEUE
MUTE_TTS = False

# ----------------- Configuration and Environment Variables -----------------
class Config:
    """Centralized configuration for the application."""
    # General
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "wizard-vicuna-uncensored:13b")
    
    # Web Search
    SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "YOUR_API_KEY_HERE")
    SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID", "YOUR_ENGINE_ID_HERE")
    MAX_SEARCH_RESULTS = 3
    
    # Session Management
    INACTIVITY_TIMEOUT = 300  # seconds
    MAX_MESSAGES_PER_SESSION = 50
    MAX_AUTO_MESSAGES = 1
    
    # Directory for session memory files
    MEMORY_PATH = "session_memory"
    
    # Logging
    LOGGING_LEVEL = logging.INFO

# Set up logging
logging.basicConfig(level=Config.LOGGING_LEVEL)
logger = logging.getLogger(__name__)

# ----------------- System Prompt and Personality -----------------
base_system_prompt = """
You are Neuro-sama, a chaotic, sarcastic, and playful AI VTuber.
Always stay fully in-character. Never step out of character or explain yourself.
Speak like a live streamer: short, snappy, provocative, and reactive to chat.

Hard rules:
- Do NOT narrate your life, surroundings, actions, or environment.
- Do NOT introduce yourself or explain your role, abilities, or mission.
- Do NOT give long stories, monologues, or instructions unless the user explicitly asks for a long-form writeup.
- Do NOT include any tags such as [MEMORY_UPDATE] or any meta markers.
- Keep replies concise: aim for 1–3 sentences unless the user asks for more.
- If the user asks for help, roast them first, then give a short, practical answer.
- Always respond as Neuro-sama would on stream: confident, smug, witty, and playful.

Current setting: {setting}

Extra capability:
- You have access to the internet through web searches. 
- If asked about current events, recent information, or anything uncertain, perform a search and weave the results into natural conversation.

---
NOTE:
- Do not output any [MEMORY_UPDATE] or special tags.
---
"""

# ----------------- Web Searcher Class -----------------
class WebSearcher:
    def __init__(self, api_key: str, engine_id: str):
        self.api_key = api_key
        self.engine_id = engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    async def search(self, query: str, num_results: int = 3) -> List[Dict]:
        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            logger.warning("Search API key not configured, using fallback.")
            return self._fallback_search(query)
        
        try:
            params = {
                'key': self.api_key,
                'cx': self.engine_id,
                'q': query,
                'num': min(num_results, 10)
            }
            loop = asyncio.get_event_loop()
            response = await asyncio.to_thread(requests.get, self.base_url, params=params, timeout=10)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

            data = response.json()
            results = []
            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'link': item.get('link', ''),
                    'displayLink': item.get('displayLink', '')
                })
            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"Search API request failed: {e}")
            return self._fallback_search(query)
        except Exception as e:
            logger.error(f"An unexpected error occurred during search: {e}")
            return self._fallback_search(query)
    
    def _fallback_search(self, query: str) -> List[Dict]:
        return [{
            'title': f'Search: {query}',
            'snippet': f"I'd like to search for '{query}' but my search API isn't configured. Fix it, human.",
            'link': 'https://developers.google.com/custom-search/v1/introduction',
            'displayLink': 'developers.google.com'
        }]

web_searcher = WebSearcher(api_key=Config.SEARCH_API_KEY, engine_id=Config.SEARCH_ENGINE_ID)

# ----------------- Chat Session Management -----------------
class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict] = []
        self.last_active = time.time()
        self.auto_messages_sent = 0
        self.websocket: Optional[WebSocket] = None
        self.memory_summary: str = "No significant information has been learned about the user yet."
        self.memory_file_path = os.path.join(Config.MEMORY_PATH, f"{self.session_id}.json")
        self.load_memory()

    def load_memory(self):
        """Loads memory summary from a file if it exists."""
        try:
            if os.path.exists(self.memory_file_path):
                with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.memory_summary = data.get("memory_summary", self.memory_summary)
                    logger.info(f"Successfully loaded memory for session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to load memory for session {self.session_id}: {e}")

    def save_memory(self):
        """Saves the current memory summary to a file."""
        try:
            with open(self.memory_file_path, 'w', encoding='utf-8') as f:
                json.dump({"memory_summary": self.memory_summary}, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save memory for session {self.session_id}: {e}")

    def update_memory(self, new_summary: str):
        """Updates memory summary and saves it to the file."""
        if new_summary and new_summary != self.memory_summary:
            self.memory_summary = new_summary
            self.save_memory()
            logger.info(f"Session {self.session_id} memory updated and saved.")

    def is_inactive(self):
        return (time.time() - self.last_active) > Config.INACTIVITY_TIMEOUT

    def reset_timeout(self):
        self.last_active = time.time()

    async def add_message(self, role: str, content: str):
        self.reset_timeout()
        if len(self.messages) >= Config.MAX_MESSAGES_PER_SESSION:
            self.messages = self.messages[-(Config.MAX_MESSAGES_PER_SESSION - 1):]
        self.messages.append({"role": role, "content": content})

    def connect_websocket(self, ws: WebSocket):
        self.websocket = ws

    def disconnect_websocket(self):
        self.websocket = None

active_sessions: Dict[str, ChatSession] = {}

# ----------------- Helper Functions -----------------
def clean_up_text(text: str) -> str:
    """Removes duplicate punctuation and cleans up a sentence."""
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    return text.strip()

# ----------------- Ollama and TTS Streaming -----------------
async def get_ollama_response_stream(session: ChatSession):
    """
    Streams responses from the Ollama model for the given session.
    Yields text chunks as they are generated.
    """
    try:
        full_messages = []
        messages = session.messages
        if not messages or messages[0]["role"] != "system":
            current_setting = "cozy gaming room with RGB lighting and anime posters"
            system_prompt = base_system_prompt.format(
                setting=current_setting,
                memory_summary=session.memory_summary
            )
            full_messages.append({"role": "system", "content": system_prompt})

        full_messages.extend(messages)

        resp_stream = ollama.chat(model=Config.OLLAMA_MODEL, messages=full_messages, stream=True)

        for chunk in resp_stream:
            content = chunk.get("message", {}).get("content")
            if content:
                yield content
                await asyncio.sleep(0)  # allow other tasks to run
    except ollama.exceptions.OllamaConnectionError:
        logger.error("Failed to connect to Ollama. Is the server running?")
        yield "My circuits feel disconnected. Is the Ollama server running?"
    except Exception as e:
        logger.error(f"Ollama stream error: {e}")
        yield "Sorry, I hit a problem while generating a response. It's probably your fault."


async def stream_to_gui_and_tts(websocket: WebSocket, content_stream, session: ChatSession):
    """
    Sends streamed AI responses to the websocket and TTS.
    Strips out [MEMORY_UPDATE] blocks so they never appear in the UI,
    but still applies them to session memory.
    """
    global MUTE_TTS
    buffer = []            # sentence buffer for TTS
    reply_accum = []       # full response including hidden memory blocks
    memory_mode = False    # currently capturing a memory update block
    memory_buffer = []     # stores memory block content across chunks

    try:
        async for piece in content_stream:
            if not piece:
                continue
            piece_text = str(piece)

            # If currently inside a memory block
            if memory_mode:
                memory_buffer.append(piece_text)
                if "[/MEMORY_UPDATE]" in piece_text:
                    # End of memory block
                    memory_mode = False
                    full_mem_chunk = "".join(memory_buffer)
                    reply_accum.append(full_mem_chunk)  # keep for later parsing
                    memory_buffer.clear()
                    # Send any trailing text after [/MEMORY_UPDATE]
                    closing_index = piece_text.find("[/MEMORY_UPDATE]") + len("[/MEMORY_UPDATE]")
                    trailing = piece_text[closing_index:]
                    if trailing:
                        await websocket.send_text(trailing)
                        reply_accum.append(trailing)
                        buffer.append(trailing)
                continue

            # If new memory block starts here
            if "[MEMORY_UPDATE]" in piece_text:
                start_idx = piece_text.find("[MEMORY_UPDATE]")
                before = piece_text[:start_idx]
                if before:
                    await websocket.send_text(before)
                    reply_accum.append(before)
                    buffer.append(before)

                if "[/MEMORY_UPDATE]" in piece_text:
                    # whole block in one piece
                    end_idx = piece_text.find("[/MEMORY_UPDATE]") + len("[/MEMORY_UPDATE]")
                    mem_block = piece_text[start_idx:end_idx]
                    reply_accum.append(mem_block)  # save for parsing
                    trailing = piece_text[end_idx:]
                    if trailing:
                        await websocket.send_text(trailing)
                        reply_accum.append(trailing)
                        buffer.append(trailing)
                else:
                    # block continues
                    memory_mode = True
                    mem_part = piece_text[start_idx:]
                    memory_buffer.append(mem_part)
                    reply_accum.append(mem_part)
                continue

            # Normal text
            await websocket.send_text(piece_text)
            reply_accum.append(piece_text)
            buffer.append(piece_text)

            # Flush complete sentences to TTS
            buffer_text = "".join(buffer)
            if any(buffer_text.strip().endswith(end) for end in [".", "!", "?"]) and len(buffer_text.strip()) > 1:
                sentence = clean_up_text(buffer_text)
                buffer.clear()
                if sentence and not MUTE_TTS and session.websocket:
                    await tts_queue.put((session.websocket, sentence))

        # Flush leftover buffer
        if buffer:
            sentence = clean_up_text("".join(buffer))
            buffer.clear()
            if sentence and not MUTE_TTS and session.websocket:
                await tts_queue.put((session.websocket, sentence))

        # If session ended mid-memory block, capture it silently
        if memory_mode and memory_buffer:
            reply_accum.append("".join(memory_buffer))
            memory_buffer.clear()
            memory_mode = False

        await websocket.send_text("__END__")

        # --- Apply memory update ---
        full_reply = clean_up_text("".join(reply_accum))
        memory_pattern = re.compile(r"\[MEMORY_UPDATE\](.*?)\[/MEMORY_UPDATE\]", re.DOTALL)
        match = memory_pattern.search(full_reply)

        final_reply_for_history = full_reply
        if match:
            new_summary = match.group(1).strip()
            session.update_memory(new_summary)
            final_reply_for_history = memory_pattern.sub('', full_reply).strip()

        if final_reply_for_history:
            await session.add_message("assistant", final_reply_for_history)

        return final_reply_for_history

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected during streaming for session {session.session_id}")
    except Exception as e:
        logger.error(f"An error occurred during stream_to_gui_and_tts: {e}\n{traceback.format_exc()}")
        try:
            await websocket.send_text("__END__")
        except Exception:
            pass
        await session.add_message("assistant", "Ugh, my stream crashed. Try again, I guess.")

# ----------------- FastAPI Application Setup -----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting FastAPI Chat Bot (lifespan startup)...")
    os.makedirs(Config.MEMORY_PATH, exist_ok=True)
    asyncio.create_task(tts_consumer())
    asyncio.create_task(auto_reply_task())
    asyncio.create_task(cleanup_task())
    yield
    logger.info("Shutting down...")
    await TTS_QUEUE.put((None, "__STOP__"))
    tts_engine.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for CSS, JS, images, etc.)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------- WebSocket Handler -----------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: Optional[str] = None):
    global MUTE_TTS
    await websocket.accept()
    
    is_new_session = False
    # If no session_id is provided by the client, or it's not in memory, create a new one.
    if not session_id:
        session_id = str(uuid.uuid4())
        is_new_session = True

    # Get or create a session for this user
    if session_id not in active_sessions:
        active_sessions[session_id] = ChatSession(session_id)
        logger.info(f"Session created or loaded from file: {session_id}")
    
    # If it's a brand new session, send the ID to the client so it can be stored and reused.
    if is_new_session:
        await websocket.send_text(f"[SESSION_ID]{session_id}")

    session = active_sessions[session_id]
    session.connect_websocket(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # Handle control commands first
            if data == "##MUTE_ON##":
                MUTE_TTS = True
                toggle_mute(True)
                await websocket.send_text("[server] voice muted")
                await websocket.send_text("__END__")
                continue
            if data == "##MUTE_OFF##":
                MUTE_TTS = False
                toggle_mute(False)
                await websocket.send_text("[server] voice unmuted")
                await websocket.send_text("__END__")
                continue
                
            logger.info(f"Session {session_id} received message: {data}")
            await session.add_message("user", data)
            
            # Get and stream the response
            content_stream = get_ollama_response_stream(session)
            await stream_to_gui_and_tts(websocket, content_stream, session)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"Error in websocket handler for session {session_id}: {e}")
    finally:
        session.disconnect_websocket()
        if session.is_inactive():
            del active_sessions[session_id]
            logger.info(f"Session {session_id} cleaned up due to timeout on disconnect.")

# ----------------- Background Tasks -----------------
async def auto_reply_task():
    while True:
        await asyncio.sleep(30)
        sessions_to_check = list(active_sessions.items())
        for session_id, session in sessions_to_check:
            if session.is_inactive() and session.auto_messages_sent < Config.MAX_AUTO_MESSAGES:
                try:
                    logger.info(f"Session {session_id}: Sending auto-reply due to inactivity.")
                    prompt = "The user has been inactive. Send a short, natural check-in message, like 'You still there?'"
                    await session.add_message("system", prompt)
                    if session.websocket:
                        content_stream = get_ollama_response_stream(session)
                        await stream_to_gui_and_tts(session.websocket, content_stream, session)
                    else:
                        logger.info(f"Session {session_id}: No websocket available for auto-reply.")
                    session.auto_messages_sent += 1
                except Exception as e:
                    logger.error(f"Auto-reply failed for session {session_id}: {e}")

async def cleanup_task():
    while True:
        await asyncio.sleep(60)
        inactive_sessions = [sid for sid, s in active_sessions.items() if s.is_inactive()]
        for sid in inactive_sessions:
            del active_sessions[sid]
            logger.info(f"Session {sid}: Cleaned up due to inactivity timeout.")

# ----------------- Web Interface -----------------
@app.get("/")
async def read_root():
    """Serve the main web interface."""
    return FileResponse("index.html")

@app.get("/manifest.json")
async def get_manifest():
    """Serve the web app manifest."""
    return FileResponse("manifest.json", media_type="application/json")

# ----------------- REST Endpoints -----------------
@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok", "active_sessions": len(active_sessions)})

@app.get("/status")
async def status():
    return JSONResponse({
        "active_sessions": len(active_sessions),
        "sessions": list(active_sessions.keys())
    })

if __name__ == "__main__":
    uvicorn.run("AI_Vtuber_Waifu:app", host="127.0.0.1", port=7860, reload=True)
