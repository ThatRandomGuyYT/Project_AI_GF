import sys
import asyncio
import websockets
import requests
import json
from datetime import datetime
from typing import Optional
import ssl

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QScrollArea, QFrame,
    QFileDialog, QMessageBox, QGraphicsDropShadowEffect, QSizePolicy,
    QSpacerItem
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve,
    QRect, QSize
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette
)

# Configuration
WS_URL = "ws://127.0.0.1:7860/ws"
HEALTH_URL = "http://127.0.0.1:7860/health"

class ConnectionWorker(QThread):
    """Robust WebSocket connection worker with better error handling"""

    message_received = pyqtSignal(str, bool)
    connection_changed = pyqtSignal(bool, str)
    location_changed = pyqtSignal(str)

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.should_reconnect = True
        self.max_retries = 10
        self._pending_messages = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.message_queue: Optional[asyncio.Queue] = None
        self._ws = None
        self.retry_delays = [1, 2, 3, 5, 5, 10, 10, 15, 20, 30]  # Progressive backoff

    def run(self):
        # Create and run an event loop inside this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.message_queue = asyncio.Queue()
        try:
            self.loop.run_until_complete(self._connection_loop())
        except Exception as e:
            self.connection_changed.emit(False, f"Connection loop error: {e}")
        finally:
            self._cleanup_loop()

    def _cleanup_loop(self):
        """Clean shutdown of asyncio loop"""
        try:
            # Cancel all tasks
            tasks = asyncio.all_tasks(loop=self.loop)
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete cancellation
            if tasks:
                self.loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                
            # Shutdown async generators
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        except Exception as e:
            print(f"Error during loop cleanup: {e}")
        finally:
            self.loop.close()

    async def _connection_loop(self):
        retry_count = 0
        
        while self.should_reconnect and retry_count < self.max_retries:
            try:
                self.connection_changed.emit(False, f"Connecting... (attempt {retry_count + 1}/{self.max_retries})")
                
                # Add extra headers for WebSocket handshake
                extra_headers = {
                    "Origin": "http://127.0.0.1:7860",
                    "User-Agent": "ModernChatGUI/2.1"
                }
                
                # Try connection with timeout and proper error handling
                async with websockets.connect(
                    self.url,
                    ping_timeout=30,
                    ping_interval=20,
                    close_timeout=10,
                    extra_headers=extra_headers,
                    # Disable SSL verification for local connections
                    ssl=None if self.url.startswith("ws://") else ssl.create_default_context()
                ) as websocket:
                    self._ws = websocket
                    retry_count = 0  # Reset retry count on successful connection
                    self.connection_changed.emit(True, "Connected successfully")

                    # Send any pending messages
                    while self._pending_messages:
                        msg = self._pending_messages.pop(0)
                        await self.message_queue.put(msg)

                    # Create tasks for sending and receiving
                    receive_task = asyncio.create_task(self._receive_messages(websocket))
                    send_task = asyncio.create_task(self._send_messages(websocket))

                    try:
                        # Wait for either task to complete/fail
                        done, pending = await asyncio.wait(
                            [receive_task, send_task], 
                            return_when=asyncio.FIRST_EXCEPTION
                        )

                        # Cancel remaining tasks
                        for task in pending:
                            task.cancel()
                            
                        # Check if any task raised an exception
                        for task in done:
                            if task.exception():
                                raise task.exception()
                                
                    except asyncio.CancelledError:
                        # Clean cancellation
                        pass
                    except Exception as e:
                        self.connection_changed.emit(False, f"Connection error: {e}")
                        raise

            except websockets.exceptions.InvalidStatusCode as e:
                if "403" in str(e) or "Forbidden" in str(e):
                    self.connection_changed.emit(False, "403 Forbidden - Server rejected connection. Check CORS settings.")
                else:
                    self.connection_changed.emit(False, f"Connection error: {e}")
                retry_count += 1
                
            except websockets.exceptions.ConnectionClosedError as e:
                self.connection_changed.emit(False, f"Connection closed: {e}")
                retry_count += 1
                
            except OSError as e:
                self.connection_changed.emit(False, f"Network error: {e}")
                retry_count += 1
                
            except Exception as e:
                self.connection_changed.emit(False, f"Unexpected error: {e}")
                retry_count += 1

            # Wait before retrying (with progressive backoff)
            if self.should_reconnect and retry_count < self.max_retries:
                delay = self.retry_delays[min(retry_count - 1, len(self.retry_delays) - 1)]
                self.connection_changed.emit(False, f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

        if retry_count >= self.max_retries:
            self.connection_changed.emit(False, "Max retries exceeded. Check server status.")

    async def _receive_messages(self, websocket):
        """Handle incoming WebSocket messages"""
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Skip binary messages for now
                    continue

                # Handle special messages
                if message.startswith("__LOCATION__"):
                    location = message[len("__LOCATION__"):]
                    self.location_changed.emit(location)
                    continue

                # Handle end-of-message marker
                if message == "__END__":
                    self.message_received.emit("", True)  # Signal message complete
                else:
                    self.message_received.emit(message, False)  # Streaming content
                    
        except websockets.exceptions.ConnectionClosedOK:
            self.connection_changed.emit(False, "Connection closed normally")
        except websockets.exceptions.ConnectionClosedError as e:
            self.connection_changed.emit(False, f"Connection lost: {e}")
            raise
        except Exception as e:
            self.connection_changed.emit(False, f"Receive error: {e}")
            raise

    async def _send_messages(self, websocket):
        """Handle outgoing WebSocket messages"""
        try:
            while True:
                message = await self.message_queue.get()
                await websocket.send(message)
                self.message_queue.task_done()
        except websockets.exceptions.ConnectionClosedOK:
            pass
        except websockets.exceptions.ConnectionClosedError as e:
            self.connection_changed.emit(False, f"Send failed - connection lost: {e}")
            raise
        except Exception as e:
            self.connection_changed.emit(False, f"Send error: {e}")
            raise

    def send_message(self, message: str):
        """Queue a message to be sent"""
        if self.loop and self.loop.is_running() and not self.loop.is_closed():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.message_queue.put(message), self.loop
                )
                # Don't block - if it fails, add to pending
                try:
                    future.result(timeout=0.1)
                except Exception:
                    self._pending_messages.append(message)
            except Exception:
                self._pending_messages.append(message)
        else:
            self._pending_messages.append(message)

    def stop(self):
        """Stop the connection worker gracefully"""
        self.should_reconnect = False
        
        if self.loop and not self.loop.is_closed():
            try:
                # Schedule websocket closure
                if self._ws:
                    asyncio.run_coroutine_threadsafe(self._close_websocket(), self.loop)
                
                # Stop the event loop
                self.loop.call_soon_threadsafe(self.loop.stop)
            except Exception as e:
                print(f"Error stopping worker: {e}")
        
        # Wait for thread to finish
        if self.isRunning():
            self.quit()
            if not self.wait(5000):  # 5 second timeout
                print("Warning: Worker thread did not stop cleanly")

    async def _close_websocket(self):
        """Close WebSocket connection gracefully"""
        try:
            if self._ws and not self._ws.closed:
                await self._ws.close()
        except Exception as e:
            print(f"Error closing websocket: {e}")


class ModernButton(QPushButton):
    """Custom button with modern styling and hover effects"""

    def __init__(self, text: str = "", icon_text: str = "", primary: bool = False):
        super().__init__()
        self.setText(f"{icon_text} {text}".strip())
        self.primary = primary
        self.setMinimumHeight(44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        if primary:
            self.setStyleSheet(
                """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                        stop:0 #667eea, stop:1 #764ba2);
                    color: white;
                    border: none;
                    border-radius: 22px;
                    font-size: 14px;
                    font-weight: 600;
                    padding: 0 20px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                        stop:0 #7c8df0, stop:1 #8a5fb8);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                        stop:0 #5a6fd8, stop:1 #6a4190);
                }
                QPushButton:disabled {
                    background: #4a4a4a;
                    color: #888888;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QPushButton {
                    background: rgba(255, 255, 255, 0.03);
                    color: #e0e0e0;
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    border-radius: 22px;
                    font-size: 14px;
                    font-weight: 500;
                    padding: 0 16px;
                }
                QPushButton:hover {
                    background: rgba(255, 255, 255, 0.06);
                    border-color: rgba(255, 255, 255, 0.12);
                    color: white;
                }
                QPushButton:pressed {
                    background: rgba(255, 255, 255, 0.04);
                }
                """
            )

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 90))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)


class MessageBubble(QFrame):
    """Modern chat bubble with safe animation and streaming append"""

    def __init__(self, message: str, is_user: bool = False, timestamp: Optional[str] = None):
        super().__init__()
        self.message = message or ""
        self.is_user = is_user
        self.timestamp = timestamp or datetime.now().strftime("%H:%M")

        self.setMaximumWidth(640)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Container frame for bubble styling
        self.container = QFrame()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(14, 10, 14, 10)
        self.container_layout.setSpacing(6)

        # Message label
        self.message_label = QLabel(self.message)
        self.message_label.setWordWrap(True)
        font = QFont()
        font.setPointSize(13)
        self.message_label.setFont(font)
        self.message_label.setStyleSheet("color: white; background: transparent;")
        self.container_layout.addWidget(self.message_label)

        # Timestamp
        self.time_label = QLabel(self.timestamp)
        tfont = QFont()
        tfont.setPointSize(9)
        self.time_label.setFont(tfont)
        self.time_label.setStyleSheet("color: rgba(255,255,255,0.55); background: transparent;")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight if is_user else Qt.AlignmentFlag.AlignLeft)
        self.container_layout.addWidget(self.time_label)

        # Apply styles
        self.container.setStyleSheet(self._get_bubble_style())
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(0, 0, 0, 40))
        shadow.setOffset(0, 2)
        self.container.setGraphicsEffect(shadow)

        layout.addWidget(self.container)

    def _get_bubble_style(self) -> str:
        if self.is_user:
            return (
                "QFrame { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #667eea, stop:1 #764ba2);"
                "border-radius: 14px; }")
        else:
            return (
                "QFrame { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #2d3748, stop:1 #4a5568);"
                "border-radius: 14px; border: 1px solid rgba(255,255,255,0.04);}"
            )

    def append_text(self, text: str):
        """Append text for streaming messages"""
        self.message += text
        self.message_label.setText(self.message)

    def animate_in(self):
        """Animate bubble appearance"""
        content_height = self.container.sizeHint().height() + 10
        start = 0
        end = max(40, content_height)
        self.setMaximumHeight(start)
        self._anim = QPropertyAnimation(self, b"maximumHeight")
        self._anim.setDuration(280)
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.start()


class TypingIndicator(QFrame):
    """Animated typing indicator"""

    def __init__(self):
        super().__init__()
        self.setFixedSize(72, 40)
        self.setStyleSheet("QFrame { background: rgba(255,255,255,0.02); border-radius: 20px; }")

        self.dots = [QLabel("•"), QLabel("•"), QLabel("•")]
        for dot in self.dots:
            dot.setStyleSheet("color: rgba(255,255,255,0.35); font-size: 18px;")
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        for dot in self.dots:
            layout.addWidget(dot)

        self._index = 0
        self._timer = QTimer(self)
        self._timer.setInterval(220)
        self._timer.timeout.connect(self._animate)
        self._timer.start()

    def _animate(self):
        for i, dot in enumerate(self.dots):
            if i == self._index:
                dot.setStyleSheet("color: #667eea; font-size: 18px;")
            else:
                dot.setStyleSheet("color: rgba(255,255,255,0.35); font-size: 18px;")
        self._index = (self._index + 1) % len(self.dots)


class ChatArea(QScrollArea):
    """Modern chat area with message management"""

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(18, 18, 18, 18)
        self.layout.setSpacing(12)
        self.layout.addStretch()
        self.setWidget(self.container)

        self.typing_indicator = None
        self.current_ai_bubble: Optional[MessageBubble] = None

    def add_message(self, text: str, is_user: bool = False, animate: bool = True) -> MessageBubble:
        bubble = MessageBubble(text, is_user)

        wrapper = QWidget()
        wrapper.setStyleSheet('background: transparent;')
        hl = QHBoxLayout(wrapper)
        hl.setContentsMargins(0, 0, 0, 0)

        if is_user:
            hl.addStretch()
            hl.addWidget(bubble)
        else:
            hl.addWidget(bubble)
            hl.addStretch()

        # Insert before the stretch
        self.layout.insertWidget(self.layout.count() - 1, wrapper)

        if animate:
            bubble.animate_in()

        if not is_user:
            self.current_ai_bubble = bubble

        QTimer.singleShot(40, self.scroll_to_bottom)
        return bubble

    def add_system_message(self, text: str):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            "color: rgba(255,255,255,0.6); background: rgba(255,255,255,0.02); "
            "padding:8px 12px; border-radius:10px;"
        )
        self.layout.insertWidget(self.layout.count() - 1, lbl)
        QTimer.singleShot(40, self.scroll_to_bottom)

    def show_typing_indicator(self):
        if self.typing_indicator:
            return
        self.typing_indicator = TypingIndicator()
        wrapper = QWidget()
        wrapper.setStyleSheet('background: transparent;')
        hl = QHBoxLayout(wrapper)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.typing_indicator)
        hl.addStretch()
        self.layout.insertWidget(self.layout.count() - 1, wrapper)
        QTimer.singleShot(40, self.scroll_to_bottom)

    def hide_typing_indicator(self):
        if not self.typing_indicator:
            return
        # Remove the widget that contains the typing indicator
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            if item and item.widget():
                w = item.widget()
                if w.findChild(TypingIndicator):
                    self.layout.removeWidget(w)
                    w.deleteLater()
                    break
        self.typing_indicator = None

    def start_ai_message(self) -> MessageBubble:
        self.hide_typing_indicator()
        self.current_ai_bubble = self.add_message("", is_user=False, animate=False)
        return self.current_ai_bubble

    def append_to_current_message(self, text: str):
        if self.current_ai_bubble:
            self.current_ai_bubble.append_text(text)
            QTimer.singleShot(10, self.scroll_to_bottom)

    def finish_ai_message(self):
        self.current_ai_bubble = None
        QTimer.singleShot(40, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


class StatusBar(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(48)
        self.setStyleSheet("background: rgba(255,255,255,0.02); border-top:1px solid rgba(255,255,255,0.03);")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 6, 14, 6)

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #ff6b6b; font-size: 12px;")
        self.status_label = QLabel("Connecting...")
        self.status_label.setStyleSheet("color: rgba(255,255,255,0.75);")

        layout.addWidget(self.status_dot)
        layout.addWidget(self.status_label)
        layout.addStretch()

    def set_status(self, status: str, connected: bool = False):
        self.status_label.setText(status)
        color = '#51cf66' if connected else '#ff6b6b'
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 12px;")


class ModernInput(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(92)
        self.setStyleSheet(
            "background: rgba(255,255,255,0.02); border-radius: 14px; padding: 12px;"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type your message...")
        tf = QFont()
        tf.setPointSize(13)
        self.text_input.setFont(tf)
        self.text_input.setStyleSheet(
            "QLineEdit { background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.03);"
            "border-radius: 18px; padding: 12px; color: white;} QLineEdit:focus { border-color: #667eea; }"
        )

        self.send_button = ModernButton("Send", "→", primary=True)
        self.send_button.setFixedSize(100, 48)

        self.save_button = ModernButton("", "💾")
        self.save_button.setFixedSize(48, 48)
        self.save_button.setToolTip("Save Chat")

        self.health_button = ModernButton("", "📊")
        self.health_button.setFixedSize(48, 48)
        self.health_button.setToolTip("Health Check")

        layout.addWidget(self.text_input)
        layout.addWidget(self.send_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.health_button)


class ModernChatGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self._current_message_started = False
        self.setup_ui()
        self.setup_connections()
        self.setup_worker()

    def setup_ui(self):
        self.setWindowTitle("Modern Chat Bot")
        self.setMinimumSize(920, 680)
        self.resize(1024, 780)

        # Central layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # Header
        header = QFrame()
        header.setFixedHeight(72)
        header.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0f1724, stop:1 #1b1f2b); border-radius:10px;"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 8, 16, 8)
        title = QLabel("Modern Chat Bot")
        tfont = QFont()
        tfont.setPointSize(18)
        tfont.setBold(True)
        title.setFont(tfont)
        title.setStyleSheet("color: white;")
        hl.addWidget(title)
        hl.addStretch()

        main_layout.addWidget(header)

        # Chat area
        self.chat_area = ChatArea()
        main_layout.addWidget(self.chat_area)

        # Input area
        self.input_area = ModernInput()
        main_layout.addWidget(self.input_area)

        # Status bar
        self.status_bar = StatusBar()
        main_layout.addWidget(self.status_bar)

        self.chat_area.add_system_message("Welcome — Modern Chat Bot ready")

    def setup_connections(self):
        self.input_area.send_button.clicked.connect(self.send_message)
        self.input_area.text_input.returnPressed.connect(self.send_message)
        self.input_area.save_button.clicked.connect(self.save_chat)
        self.input_area.health_button.clicked.connect(self.check_health)

    def setup_worker(self):
        self.worker = ConnectionWorker(WS_URL)
        self.worker.message_received.connect(self.on_message_received)
        self.worker.connection_changed.connect(self.on_connection_changed)
        self.worker.start()
        self.status_bar.set_status("Connecting...", False)

    def send_message(self):
        text = self.input_area.text_input.text().strip()
        if not text:
            return
        
        self.chat_area.add_message(text, is_user=True)
        self.input_area.text_input.clear()
        self.chat_area.show_typing_indicator()
        self.worker.send_message(text)
        self.input_area.send_button.setEnabled(False)

    def on_message_received(self, message: str, is_complete: bool):
        # Start streaming message if not started
        if not getattr(self, '_current_message_started', False):
            self.chat_area.start_ai_message()
            self._current_message_started = True

        if message:
            self.chat_area.append_to_current_message(message)

        if is_complete:
            self.chat_area.finish_ai_message()
            self.input_area.send_button.setEnabled(True)
            self._current_message_started = False

    def on_connection_changed(self, connected: bool, status: str):
        self.status_bar.set_status(status, connected)
        if connected:
            self.chat_area.add_system_message("Connected to chat bot")
        else:
            self.chat_area.add_system_message(f"Connection issue: {status}")
            # Reset message state on disconnection
            self._current_message_started = False
            self.input_area.send_button.setEnabled(True)

    def save_chat(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Chat", f"chat_{timestamp}.txt", "Text Files (*.txt)"
        )
        if not filename:
            return
        
        try:
            messages = []
            # Iterate through layout widgets and gather MessageBubble children
            for i in range(self.chat_area.layout.count()):
                item = self.chat_area.layout.itemAt(i)
                if item and item.widget():
                    w = item.widget()
                    bubbles = w.findChildren(MessageBubble)
                    for b in bubbles:
                        sender = "You" if b.is_user else "Assistant"
                        messages.append(f"[{b.timestamp}] {sender}: {b.message}")

            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(messages))

            self.show_message("Success", f"Chat saved to {filename}")
        except Exception as e:
            self.show_message("Error", f"Failed to save chat: {e}", error=True)

    def check_health(self):
        try:
            resp = requests.get(HEALTH_URL, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                status_text = (
                    f"Status: {data.get('status', 'Unknown')}\n"
                    f"Active Sessions: {data.get('active_sessions', 'Unknown')}\n"
                    f"WebSocket Endpoint: {data.get('websocket_endpoint', '/ws')}\n"
                    f"Test Page: {data.get('test_page', '/')}"
                )
                self.show_message("Health Check", status_text)
            else:
                self.show_message("Health Check", f"Server returned {resp.status_code}", error=True)
        except requests.exceptions.ConnectionError:
            self.show_message("Health Check", "Cannot connect to server. Is it running?", error=True)
        except requests.exceptions.Timeout:
            self.show_message("Health Check", "Server request timed out", error=True)
        except Exception as e:
            self.show_message("Health Check", f"Error: {e}", error=True)

    def show_message(self, title: str, message: str, error: bool = False):
        mb = QMessageBox(self)
        mb.setWindowTitle(title)
        mb.setText(message)
        mb.setIcon(QMessageBox.Icon.Critical if error else QMessageBox.Icon.Information)
        mb.exec()

    def closeEvent(self, event):
        """Clean shutdown when closing the application"""
        try:
            if hasattr(self, 'worker') and self.worker:
                self.worker.stop()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Modern Chat Bot")
    app.setApplicationVersion("2.1")

    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(11, 14, 29))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(18, 20, 36))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(28, 30, 48))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(28, 30, 48))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = ModernChatGUI()
    window.show()

    # Center window on screen
    screen = app.primaryScreen().availableGeometry()
    window.move(
        (screen.width() - window.width()) // 2, 
        (screen.height() - window.height()) // 2
    )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()