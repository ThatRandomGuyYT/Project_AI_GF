# AI VTuber Waifu

This project is a comprehensive AI VTuber application that combines a chat interface, a text-to-speech (TTS) engine with voice cloning, and a 3D model viewer. It's designed to provide a complete VTuber experience, allowing users to interact with an AI personality, hear its responses in a cloned voice, and see a 3D model that can be animated.

## Features

*   **AI Chat:** A real-time chat interface for interacting with an AI VTuber powered by the Ollama language model.
*   **Text-to-Speech (TTS):** A sophisticated TTS engine that converts the AI's responses into speech.
*   **Voice Cloning:** The ability to clone a voice from an audio file, giving the AI a unique and customizable voice.
*   **3D Model Viewer:** A GPU-accelerated 3D model viewer that can load and animate 3D models in the GLB format.
*   **Modern GUI:** A user-friendly interface built with PyQt6, providing a seamless and engaging user experience.

## Components

### AI_Vtuber_Waifu.py

This is the main backend of the application, built with FastAPI. It handles the following:

*   **WebSocket Communication:** Manages the real-time chat between the user and the AI.
*   **AI Chat Logic:** Integrates with the Ollama language model to generate responses from the AI VTuber.
*   **Session Management:** Keeps track of chat sessions and user interactions.
*   **TTS Integration:** Sends generated text to the TTS engine for audio playback.

### gui_2.py

This file contains the user interface for the chat application, built with PyQt6. It provides a modern, chat-bubble-style interface for interacting with the AI VTuber. Key features include:

*   **WebSocket Client:** Connects to the FastAPI backend to send and receive messages.
*   **Chat Display:** Shows the conversation in a user-friendly format.
*   **User Input:** Allows the user to type and send messages to the AI.
*   **Status Indicators:** Displays the connection status and other useful information.

### tts_engine.py

The TTS engine is responsible for converting the AI's text responses into speech. It uses the Coqui TTS library and includes several advanced features:

*   **Voice Cloning:** Can clone a voice from a provided audio file to give the AI a unique voice.
*   **Caching:** Caches generated audio to reduce latency and improve performance.
*   **Asynchronous Processing:** Generates and plays audio without blocking the main application.

### vtube_studio_clone.py

This is a GPU-accelerated 3D model viewer that serves as a clone of VTube Studio. It's built with PyQt6 and OpenGL and has the following capabilities:

*   **GLB Model Loading:** Can load and display 3D models in the GLB format.
*   **GPU Acceleration:** Uses OpenGL for efficient, hardware-accelerated rendering.
*   **Animation Control:** Allows for programmatic animation of the 3D model.
*   **User Interaction:** Supports mouse controls for rotating and zooming the model.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Ollama installed and running
*   A C++ compiler (for Coqui TTS)
*   Portaudio (for audio playback)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ThatRandomGuyYT/ai-vtuber-waifu.git
    cd ai-vtuber-waifu
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Run the backend:**
    ```bash
    python AI_Vtuber_Waifu.py
    ```

2.  **Run the GUI:**
    ```bash
    python gui_2.py
    ```

3.  **Run the VTube Studio Clone (optional):**
    ```bash
    python vtube_studio_clone.py
    ```
