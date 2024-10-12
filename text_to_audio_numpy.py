import websocket
import json
import base64
import numpy as np
import librosa
import threading
import os
import queue
import time

url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
apikey = os.getenv("OPENAI_API_KEY")

class PersistentWebSocket:
    def __init__(self):
        self.ws = None
        self.message_queue = queue.Queue()
        self.audio_string = ""
        self.audio_complete = threading.Event()
        self.websocket_thread = None
        self.is_connected = threading.Event()

    def on_message(self, ws, message):
        socket_message = json.loads(message)

        message_type = socket_message["type"]

        if message_type == "response.audio.delta":
            delta = socket_message["delta"]
            self.audio_string += delta
        elif message_type == "response.done":
            self.audio_complete.set()
        elif message_type == "rate_limits.updated":
            print(message)

    def on_open(self, ws):
        print("WebSocket connection opened")
        self.is_connected.set()

    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_connected.clear()

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def connect(self):
        self.ws = websocket.WebSocketApp(
            url,
            on_message=self.on_message,
            on_open=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
            header={
                "Authorization": f"Bearer {apikey}",
                "OpenAI-Beta": "realtime=v1",
            }
        )
        self.websocket_thread = threading.Thread(target=self.ws.run_forever)
        self.websocket_thread.start()
        # Wait for the connection to be established
        self.is_connected.wait(timeout=10)
        if not self.is_connected.is_set():
            raise ConnectionError("Failed to establish WebSoc