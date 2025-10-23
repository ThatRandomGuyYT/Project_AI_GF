import asyncio
import os
import re
import shutil
import tempfile
import threading
import hashlib
import torch
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
import simpleaudio as sa
from TTS.api import TTS
from asyncio import Queue
import time
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
add_safe_globals([XttsConfig])

from TTS.tts.models.xtts import XttsAudioConfig
add_safe_globals([XttsAudioConfig])

from TTS.config.shared_configs import BaseDatasetConfig
add_safe_globals([BaseDatasetConfig])

from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict

import torch as _torch
_original_torch_load = _torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
_torch.load = _patched_torch_load

# ----------------- Configuration -----------------
# --- Core Settings
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MAX_CACHE_SIZE = 100  # Number of audio clips to cache in memory
MAX_WORKERS = 2       # Threads for audio generation/playback
# --- Behavior Settings
DEDUPLICATION_TIME = 3.0 # Seconds to ignore repeated text
PAUSE_BETWEEN_CHUNKS = 0.1 # Seconds to pause between spoken text chunks
# --- Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Queue for TTS requests: (websocket_client, text_to_speak)
TTS_QUEUE: Queue[Tuple[Optional[Any], str]] = Queue()
STOP_SENTINEL = (None, "__STOP__") # Sentinel to stop the consumer


class OptimizedTTS:
    """
    An optimized Text-to-Speech engine that uses caching, pre-fetching,
    and asynchronous operations for responsive audio playback.
    """
    def __init__(self, model_name: str = MODEL_NAME, max_cache_size: int = MAX_CACHE_SIZE):
        """
        Initializes the TTS engine, loads the model, and pre-warms it.
        """
        self.model: Optional[TTS] = None
        self.is_ready = False
        self.muted = False
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        # LRU Cache: Stores file paths to generated audio
        self.audio_cache: OrderedDict[str, str] = OrderedDict()
        self.max_cache_size = max_cache_size
        self.cache_lock = threading.Lock()
        
        # Playback control
        self._cancel_event = threading.Event()
        self._current_play_obj: Optional[sa.PlayObject] = None
        
        # Deduplication state
        self._last_spoken_key: Optional[str] = None
        self._last_spoken_at: float = 0.0
        
        self.speaker_wav: Optional[str] = None
        self._use_gpu = torch.cuda.is_available()

        self._initialize_model(model_name)

    def _initialize_model(self, model_name: str):
        """Loads the TTS model and pre-warms it in a background thread."""
        if self._use_gpu:
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}. Using GPU for TTS.")
        else:
            logger.warning("CUDA not available. Falling back to CPU for TTS, which will be slower.")
        
        # Load model in a separate thread to avoid blocking
        self.executor.submit(self._load_and_prewarm_model, model_name)

    def _load_and_prewarm_model(self, model_name: str):
        """Loads and warms up the TTS model."""
        try:
            self.model = TTS(model_name, gpu=self._use_gpu)
            logger.info(f"TTS model '{model_name}' loaded successfully.")
            
            # Pre-warm the model to reduce latency on the first synthesis
            logger.info("Pre-warming TTS model...")
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            try:
                prewarm_kwargs = {"text": "Ready to synthesize.", "file_path": tmp_file}

                if self.speaker_wav:
                    # If a clone voice is already set, use it
                    prewarm_kwargs.update({"speaker_wav": self.speaker_wav, "language": "en"})
                else:
                     # Try to use a built-in speaker if the model exposes a list
                    speakers = getattr(self.model, "speakers", None)
                    if speakers and len(speakers) > 0:
                        prewarm_kwargs.update({"speaker": speakers[0], "language": "en"})
                    else:
                        # No speaker/clone available — skip pre-warm but mark ready
                        logger.info("Skipping pre-warm: multi-speaker model but no speaker or clone set yet.")
                        self.is_ready = True
                        return

                self.model.tts_to_file(**prewarm_kwargs)
                self.is_ready = True
                logger.info("TTS model pre-warmed and ready.")
            finally:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
        except Exception:
            logger.error(f"Failed to load or pre-warm TTS model:\n{traceback.format_exc()}")
            self.is_ready = False

    def set_voice_clone(self, wav_path: str):
        """
        Sets a reference WAV file to clone a voice for synthesis.
        For best results, use a 6-30 second clean audio clip.
        """
        if not os.path.exists(wav_path):
            logger.error(f"Voice clone WAV file not found: {wav_path}")
            return
        self.speaker_wav = wav_path
        logger.info(f"XTTS voice clone reference set to: {wav_path}")

    def _clean_text(self, text: str) -> str:
        """Removes markdown and extra whitespace for cleaner synthesis."""
        text = re.sub(r'[\*`]', '', text) # Remove markdown characters
        text = re.sub(r'\s+', ' ', text)    # Collapse whitespace
        return text.strip()

    def _get_cache_key(self, text: str, speed: float, language: str) -> str:
        """Generates a unique cache key for a given text and its settings."""
        # The speaker_wav path is included to ensure different voices aren't mixed up
        voice_id = self.speaker_wav if self.speaker_wav else "default"
        payload = f"{text.strip().lower()}|{speed}|{language}|{voice_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def _generate_audio(self, text: str, speed: float, language: str) -> Optional[str]:
        """
        Generates audio for the given text, utilizing the cache.
        Returns the file path to the generated WAV file.
        """
        if not self.is_ready or not self.model:
            logger.warning("TTS model is not ready, cannot generate audio.")
            return None

        cache_key = self._get_cache_key(text, speed, language)
        
        # Check cache first
        with self.cache_lock:
            if cache_key in self.audio_cache:
                # Move to end to mark as recently used
                self.audio_cache.move_to_end(cache_key)
                cached_path = self.audio_cache[cache_key]
                if os.path.exists(cached_path):
                    # Return a copy to prevent race conditions with file deletion
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                    shutil.copy2(cached_path, temp_file)
                    return temp_file

        # If not in cache, generate audio
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            
            tts_kwargs = {
                "text": text,
                "file_path": temp_file,
                "speaker_wav": self.speaker_wav,
                "language": language,
                "speed": speed
            }

            # Run synthesis in a thread to avoid blocking the event loop
            await asyncio.to_thread(self.model.tts_to_file, **tts_kwargs)

            # Update cache
            with self.cache_lock:
                # Evict the least recently used item if cache is full
                if len(self.audio_cache) >= self.max_cache_size:
                    oldest_key, old_path = self.audio_cache.popitem(last=False)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                
                # Store a separate copy for the cache
                cache_copy = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=tempfile.gettempdir()).name
                shutil.copy2(temp_file, cache_copy)
                self.audio_cache[cache_key] = cache_copy
                
            return temp_file
        except Exception:
            logger.error(f"TTS audio generation failed:\n{traceback.format_exc()}")
            return None

    async def _play_audio_async(self, audio_path: str):
        """Plays an audio file asynchronously and handles cancellation."""
        loop = asyncio.get_running_loop()
        
        def _play():
            try:
                wave_obj = sa.WaveObject.from_wave_file(audio_path)
                self._current_play_obj = wave_obj.play()
                self._current_play_obj.wait_done()
            except Exception:
                logger.error(f"Audio playback failed:\n{traceback.format_exc()}")
            finally:
                self._current_play_obj = None

        await loop.run_in_executor(self.executor, _play)

    def _split_text(self, text: str) -> list[str]:
        """
        Splits text into manageable chunks for smoother playback, respecting
        sentence boundaries.
        """
        text = self._clean_text(text)
        if not text:
            return []
        
        # Use regex to split by sentences, keeping the delimiters
        sentences = re.split(r'([.!?]+\s*)', text)
        if len(sentences) <= 1:
            return [text] if text else []
            
        # Group sentences into chunks of a reasonable size
        chunks, current_chunk = [], ""
        for i in range(0, len(sentences), 2):
            sentence_part = sentences[i] + (sentences[i+1] if i + 1 < len(sentences) else "")
            if len(current_chunk) + len(sentence_part) > 250 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence_part
            else:
                current_chunk += sentence_part
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    async def speak(self, text: str, speed: float = 1.0, language: str = "en"):
        """
        High-level method to synthesize and speak text. It splits text into
        chunks, pre-fetches the next chunk while playing the current one,
        and handles deduplication.
        """
        if not self.is_ready or self.muted or not text.strip():
            return

        # --- Deduplication Check
        cleaned_text = self._clean_text(text)
        current_key = self._get_cache_key(cleaned_text, speed, language)
        now = time.time()
        if self._last_spoken_key == current_key and (now - self._last_spoken_at) < DEDUPLICATION_TIME:
            return
            
        self._cancel_event.clear()
        chunks = self._split_text(cleaned_text)
        if not chunks:
            return

        try:
            # --- Prefetching and Playback Loop
            # Generate the first chunk
            next_audio_task = asyncio.create_task(self._generate_audio(chunks[0], speed, language))
            
            for i in range(len(chunks)):
                if self._cancel_event.is_set():
                    break
                
                # Wait for the current audio to be ready
                audio_path = await next_audio_task
                
                # Start generating the next chunk while we play the current one
                if i + 1 < len(chunks):
                    next_audio_task = asyncio.create_task(self._generate_audio(chunks[i + 1], speed, language))
                
                if audio_path:
                    await self._play_audio_async(audio_path)
                    os.remove(audio_path) # Clean up immediately after playing
                
                if self._cancel_event.is_set():
                    break
                
                # Optional pause for more natural speech flow
                if i < len(chunks) - 1:
                    await asyncio.sleep(PAUSE_BETWEEN_CHUNKS)
            
            # --- Update deduplication state on successful completion
            self._last_spoken_key = current_key
            self._last_spoken_at = time.time()
        except Exception:
            logger.error(f"The 'speak' operation failed:\n{traceback.format_exc()}")
        finally:
             # Ensure the last pending generation task is cancelled if the loop breaks
            if 'next_audio_task' in locals() and not next_audio_task.done():
                next_audio_task.cancel()


    def cancel(self):
        """Stops the current playback immediately."""
        self._cancel_event.set()
        if self._current_play_obj and self._current_play_obj.is_playing():
            self._current_play_obj.stop()
            logger.info("Current TTS playback cancelled.")
            
    def cleanup_cache(self):
        """Deletes all cached audio files from disk."""
        with self.cache_lock:
            for path in self.audio_cache.values():
                if os.path.exists(path):
                    os.remove(path)
            self.audio_cache.clear()
        logger.info("TTS audio cache cleared.")
    def close(self):
        """Gracefully stop threads and clear cache."""
        try:
            self.cancel()
        except Exception:
            pass
        self.executor.shutdown(wait=False)
        self.cleanup_cache()

# ----------------- Singleton Instance -----------------
# This ensures only one TTS engine is active, managing resources efficiently.
tts_engine = OptimizedTTS()


# ----------------- Public API Functions -----------------
async def speak(text: str):
    """A simple public interface to make the TTS engine speak."""
    if not text or not text.strip():
        return
    await tts_engine.speak(text)

def cancel_playback():
    """A simple public interface to cancel current speech."""
    tts_engine.cancel()

def set_voice_style(wav_path: str):
    """Sets the voice style using a reference audio file."""
    tts_engine.set_voice_clone(wav_path)
    
def toggle_mute(mute: bool):
    """Enable or disable TTS playback."""
    tts_engine.muted = mute
    if mute:
        cancel_playback()
    logger.info(f"TTS muted state set to: {mute}")

# ----------------- Asynchronous Consumer -----------------
async def tts_consumer():
    """
    An async task that continuously processes TTS requests from a queue.
    This decouples the TTS logic from the request source (e.g., a web server).
    """
    logger.info("TTS consumer started. Awaiting requests...")
    while True:
        websocket, sentence = await TTS_QUEUE.get()
        
        if (websocket, sentence) == STOP_SENTINEL:
            logger.info("TTS consumer received stop signal. Shutting down.")
            break
            
        try:
            if websocket:
                try:
                    # Notify a client that a sentence is about to be spoken
                    await websocket.send_text(f"__TTS_SPEAKING__{sentence}")
                except Exception:
                    logger.warning(f"Could not send TTS sync message via websocket.")

            await speak(sentence)
        except Exception:
            logger.error(f"Error in TTS consumer loop:\n{traceback.format_exc()}")
        finally:
            TTS_QUEUE.task_done()

# ----------------- Main Execution (for testing) -----------------
async def main():
    """
    Example usage and test function. This block will only run when the
    script is executed directly.
    """
    logger.info("Running TTS Engine test...")
    
    # Wait for the model to be ready
    while not tts_engine.is_ready:
        print("Waiting for TTS model to load...")
        await asyncio.sleep(1)
        
    print("\n--- Test 1: Simple sentence ---")
    await speak("Hello, world! This is a test of the improved text-to-speech engine.")
    await asyncio.sleep(1)

    print("\n--- Test 2: Longer text with multiple sentences ---")
    long_text = "This is a longer piece of text. It has multiple sentences. The engine should split this into chunks and play them smoothly, with a natural pause in between."
    await speak(long_text)
    await asyncio.sleep(1)

    print("\n--- Test 3: Caching demonstration ---")
    print("Speaking 'Hello, world!' again. This should be much faster as it comes from cache.")
    start_time = time.time()
    await speak("Hello, world! This is a test of the improved text-to-speech engine.")
    end_time = time.time()
    print(f"Second playback took {end_time - start_time:.2f} seconds.")

    # To test voice cloning, create a file 'voice.wav' and uncomment below
    if os.path.exists("voice.wav"):
        print("\n--- Test 4: Voice Cloning ---")
        set_voice_style("voice.wav")
        await speak("This sentence should sound like the audio from the provided WAV file.")
    else:
        print("\n--- Skipping Test 4: Place a 'voice.wav' file in the directory to test voice cloning ---")
        
    print("\n--- Test complete ---")
    tts_engine.cleanup_cache()

if __name__ == "__main__":
    # In a real application, you would start the consumer and add items to the queue.
    # For this test, we'll just call the `speak` function directly.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        tts_engine.cleanup_cache()
