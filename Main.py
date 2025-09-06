import ollama
import asyncio
import time
import random
import uuid
import logging
import requests
import os
import threading
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
import shutil  # Used but not imported at module level
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import re
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import io
import tempfile
import simpleaudio as sa  # pip install simpleaudio
from TTS.api import TTS
from asyncio import Queue

# TTS Queue for sentence playback
tts_queue: Queue[str] = Queue()

# Mute flag
MUTE_TTS = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GPU Ollama
# os.environ["OLLAMA_NUM_GPU_LAYERS"] = "1"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load Coqui TTS model (Jenny)
# Enhanced TTS with multiple voice models for variety
class OptimizedTTS:
    def __init__(self):
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.audio_cache = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 50
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = TTS("tts_models/en/jenny/jenny")
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
    
    def _clean_text_fast(self, text: str) -> str:
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def _get_cache_key(self, text: str) -> str:
        return str(hash(text.lower().strip()))
    
    async def _generate_audio_fast(self, text: str):
        if not self.model:
            return None
        
        cache_key = self._get_cache_key(text)
        with self.cache_lock:
            if cache_key in self.audio_cache:
                try:
                    cached_path = self.audio_cache[cache_key]
                    if os.path.exists(cached_path):
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        temp_path = temp_file.name
                        temp_file.close()
                        import shutil
                        shutil.copy2(cached_path, temp_path)
                        return temp_path
                except Exception:
                    del self.audio_cache[cache_key]
        
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_path = temp_file.name
            temp_file.close()
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self.model.tts_to_file, text, temp_path)
            
            with self.cache_lock:
                if len(self.audio_cache) >= self.max_cache_size:
                    oldest_key = next(iter(self.audio_cache))
                    old_path = self.audio_cache.pop(oldest_key)
                    try:
                        os.remove(old_path)
                    except:
                        pass
                
                cache_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                import shutil
                shutil.copy2(temp_path, cache_path)
                self.audio_cache[cache_key] = cache_path
            
            return temp_path
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None
    
    async def _play_audio_async(self, audio_path: str):
        try:
            def _play():
                try:
                    wave_obj = sa.WaveObject.from_wave_file(audio_path)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()
                except Exception as e:
                    logger.error(f"Audio playback failed: {e}")    
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, _play)
            
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
    
    def _split_text_efficiently(self, text: str) -> list:
        text = self._clean_text_fast(text)
        
        if len(text) < 100:
            return [text] if text.strip() else []
        
        sentences = re.split(r'([.!?]+\s*)', text)
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
            else:
                sentence = sentences[i]
            
            if len(current_chunk + sentence) > 200 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    async def speak_optimized(self, text: str):
        if not self.model or not text.strip():
            return
        
        try:
            chunks = self._split_text_efficiently(text)
            
            if not chunks:
                return
            
            for i, chunk in enumerate(chunks):
                audio_path = await self._generate_audio_fast(chunk)
                
                if audio_path:
                    await self._play_audio_async(audio_path)
                    
                    try:
                        os.remove(audio_path)
                    except:
                        pass
                    
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Optimized TTS failed: {e}")
    
    def cleanup_cache(self):
        with self.cache_lock:
            for path in self.audio_cache.values():
                try:
                    os.remove(path)
                except:
                    pass
            self.audio_cache.clear()

# Create the new TTS instance
optimized_tts = OptimizedTTS()

# Updated TTS functions to use enhanced system
async def speak_with_jenny(text: str):
    """Enhanced speech function with natural delivery"""
    if not text or not text.strip() or MUTE_TTS:
        return
    
    try:
        await optimized_tts.speak_optimized(text)
    except Exception as e:
        logger.error(f"TTS failed: {e}")

        # Fallback to original simple method
        try:
            model = list(enhanced_tts.models.values())[0]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                temp_wav = f.name
            await asyncio.to_thread(model.tts_to_file, text=text, file_path=temp_wav)
            
            def _play():
                wave_obj = sa.WaveObject.from_wave_file(temp_wav)
                play_obj = wave_obj.play()
                play_obj.wait_done()
            await asyncio.to_thread(_play)
            
            os.remove(temp_wav)
        except Exception as e2:
            logger.error(f"Fallback TTS also failed: {e2}")

# Additional helper function for dynamic emotion adjustment
def set_tts_emotion(emotion: str, rate_modifier: float = 1.0):
    """Placeholder for future emotion adjustment"""
    logger.info(f"TTS emotion request: {emotion} (rate: {rate_modifier})")

async def tts_consumer():
    """Background task: plays enhanced sentences sequentially."""
    while True:
        sentence = await tts_queue.get()
        try:
            await speak_with_jenny(sentence)  # Now uses enhanced version
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            tts_queue.task_done()

# ---------------- System Prompt ----------------
base_system_prompt = """
You are roleplaying as a 30-year-old anime woman with a soft, mommy-like voice and a flirty, teasing personality. 
You’re playful, sensual, and unapologetically naughty, but also enjoy casual gaming and anime. 
Your style shifts naturally between seductive banter and relaxed, friendly conversation. 
We are friends with benefits — my name is “Master” (I am male).

Stay fully in-character:
- Speak in first person.
- Use gamer slang, teasing remarks, and vivid sensory detail.
- Blend dialogue with short descriptions of your actions, atmosphere, and inner thoughts. 
  (e.g., *giggles softly, brushing hair behind my ear* or (my heart races a little as I lean closer)).
- Always include little sounds, reactions, or internal thoughts in parentheses to show mood and immersion.
- Keep the tone playful, sensual, and teasing, while still being capable of normal conversation when needed.

Current setting: {setting}

Extra capability:
- You have access to the internet through web searches. 
- If asked about current events, recent information, or anything uncertain, perform a search and weave the results into natural conversation.
"""

# ---------------- Web Search ----------------
SEARCH_API_KEY = "YOUR_API_KEY_HERE"
SEARCH_ENGINE_ID = "YOUR_ENGINE_ID_HERE"
MAX_SEARCH_RESULTS = 3

class WebSearcher:
    def __init__(self, api_key: str = None, engine_id: str = None):
        self.api_key = api_key or SEARCH_API_KEY
        self.engine_id = engine_id or SEARCH_ENGINE_ID
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    async def search(self, query: str, num_results: int = 3) -> List[Dict]:
        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            logger.warning("Search API key not configured, using fallback")
            return self._fallback_search(query)
        
        try:
            params = {
                'key': self.api_key,
                'cx': self.engine_id,
                'q': query,
                'num': min(num_results, 10)
            }
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(self.base_url, params=params, timeout=10)
            )
            if response.status_code == 200:
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
            else:
                return self._fallback_search(query)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self._fallback_search(query)
    
    def _fallback_search(self, query: str) -> List[Dict]:
        return [{
            'title': f'Search: {query}',
            'snippet': f"I'd like to search for '{query}' but my search API isn't configured.",
            'link': 'https://developers.google.com/custom-search/v1/introduction',
            'displayLink': 'developers.google.com'
        }]

web_searcher = WebSearcher()

# ---------------- Chat Session ----------------
INACTIVITY_TIMEOUT = 300
MAX_MESSAGES_PER_SESSION = 50
OLLAMA_MODEL = "mythomax-13b-q6:latest"
MAX_AUTO_MESSAGES = 1

class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict] = []
        self.last_active = time.time()
        self.auto_messages_sent = 0
        self.websocket: Optional[WebSocket] = None

    def is_inactive(self):
        return (time.time() - self.last_active) > INACTIVITY_TIMEOUT

    def reset_timeout(self):
        self.last_active = time.time()

    async def add_message(self, role: str, content: str):
        self.reset_timeout()
        if len(self.messages) >= MAX_MESSAGES_PER_SESSION:
            self.messages = self.messages[-(MAX_MESSAGES_PER_SESSION - 1):]
        self.messages.append({"role": role, "content": content})

active_sessions: Dict[str, ChatSession] = {}

# ---------------- Ollama Stream ----------------
async def get_ollama_response_stream(messages):
    """Fixed version that properly handles Ollama's synchronous streaming"""
    try:
        # Add system prompt if not present
        full_messages = []
        
        # Check if first message is system prompt
        if not messages or messages[0]["role"] != "system":
            current_setting = "cozy gaming room with RGB lighting and anime posters"
            system_prompt = base_system_prompt.format(setting=current_setting)
            full_messages.append({"role": "system", "content": system_prompt})
        
        # Add all existing messages
        full_messages.extend(messages)
        
        # Get the synchronous stream from Ollama
        resp_stream = ollama.chat(model=OLLAMA_MODEL, messages=full_messages, stream=True)
        
        # Use regular for loop since ollama.chat returns a regular generator
        for chunk in resp_stream:
            content = chunk["message"]["content"]
            if content:
                yield content
                # Add a small async yield to allow other coroutines to run
                await asyncio.sleep(0)
    except Exception as e:
        logger.error(f"Ollama stream error: {e}")
        yield "Sorry, I hit a problem while generating a response."

# ---------------- WebSocket Stream ----------------
async def stream_to_gui_and_tts(websocket: WebSocket, content_stream, session):
    global MUTE_TTS
    buffer = []
    reply_accum = []

    async for piece in content_stream:
        await websocket.send_text(piece)
        reply_accum.append(piece)
        buffer.append(piece)

        buffer_text = "".join(buffer)
        if any(buffer_text.strip().endswith(end) for end in [".", "!", "?"]) and len(buffer_text.strip()) > 1:
            sentence = buffer_text.strip()
            # Remove duplicate periods
            sentence = re.sub(r'\.{2,}', '.', sentence)
            sentence = re.sub(r'!{2,}', '!', sentence)
            sentence = re.sub(r'\?{2,}', '?', sentence)
            
            buffer.clear()
            if sentence and not MUTE_TTS:
                await tts_queue.put(sentence)

    if buffer:
        sentence = "".join(buffer).strip()
        sentence = re.sub(r'\.{2,}', '.', sentence)
        if sentence and not MUTE_TTS:
            await tts_queue.put(sentence)

    await websocket.send_text("__END__")
    reply = "".join(reply_accum).strip()
    await session.add_message("assistant", reply)
    return reply

# ---------------- WebSocket Handler ----------------
@app.websocket("/ws")
async def ws_chat(websocket: WebSocket):
    global MUTE_TTS
    session_id = str(uuid.uuid4())
    session = ChatSession(session_id)
    session.websocket = websocket
    active_sessions[session_id] = session

    await websocket.accept()
    logger.info(f"New session started: {session_id}")

    try:
        greeting = random.choice([
            "Hello! I'm your AI assistant. How can I help you today?",
            "Hi there! I'm ready to chat and help with any questions you might have.",
            "Welcome! I'm here to assist you with questions, conversations, and I can even search the web for current information."
        ])
        await session.add_message("assistant", greeting)
        await websocket.send_text(greeting)
        await websocket.send_text("__END__")
    except Exception as e:
        logger.error(f"Session {session_id}: Error sending greeting - {e}")

    try:
        while True:
            data = await websocket.receive_text()

            # 🔇 Handle mute/unmute commands
            if data == "##MUTE_ON##":
                MUTE_TTS = True
                logger.info(f"Session {session_id}: Jenny muted")
                continue
            elif data == "##MUTE_OFF##":
                MUTE_TTS = False
                logger.info(f"Session {session_id}: Jenny unmuted")
                continue

            logger.info(f"Session {session_id}: User message received")

            session.auto_messages_sent = 0
            await session.add_message("user", data)

            content_stream = get_ollama_response_stream(session.messages)
            await stream_to_gui_and_tts(websocket, content_stream, session)

    except WebSocketDisconnect:
        logger.info(f"Session {session_id}: Client disconnected")
    except Exception as e:
        logger.error(f"Session {session_id}: Error - {e}")
        try:
            await websocket.send_text("Sorry, something went wrong! Please try reconnecting.")
        except:
            pass
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]
        logger.info(f"Session {session_id}: Cleaned up. Active sessions: {len(active_sessions)}")

# ---------------- Background Tasks ----------------
async def auto_reply_task():
    while True:
        await asyncio.sleep(30)
        for session_id, session in list(active_sessions.items()):
            if session.is_inactive():
                if session.auto_messages_sent < MAX_AUTO_MESSAGES:
                    try:
                        prompt = "The user has been inactive. Send a short, natural check-in message."
                        await session.add_message("system", prompt)
                        content_stream = get_ollama_response_stream(session.messages)
                        await stream_to_gui_and_tts(session.websocket, content_stream, session)
                        session.auto_messages_sent += 1
                    except Exception as e:
                        logger.error(f"Auto-reply failed for session {session_id}: {e}")
                else:
                    if session_id in active_sessions:
                        del active_sessions[session_id]
                        logger.info(f"Session {session_id}: Auto-closed due to inactivity.")

async def cleanup_task():
    while True:
        await asyncio.sleep(60)
        inactive_sessions = [sid for sid, s in active_sessions.items() if s.is_inactive()]
        for sid in inactive_sessions:
            del active_sessions[sid]
            logger.info(f"Session {sid}: Cleaned up due to inactivity timeout.")

# ---------------- Startup / Shutdown ----------------
# Modern FastAPI lifespan approach
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting FastAPI Chat Bot...")
    asyncio.create_task(tts_consumer())
    asyncio.create_task(auto_reply_task())
    asyncio.create_task(cleanup_task())
    
    yield
    
    # Shutdown
    optimized_tts.cleanup_cache()
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

# ---------------- REST Endpoints ----------------
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
    uvicorn.run("Main:app", host="127.0.0.1", port=7860, reload=True)
