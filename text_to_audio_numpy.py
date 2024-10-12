import websocket
import json
import base64
import numpy as np
import librosa
import threading
import os
import queue

url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
apikey = os.getenv("OPENAI_API_KEY")

class PersistentWebSocket:
    def __init__(self):
        self.ws = None
        self.message_queue = queue.Queue()
        self.audio_string = ""
        self.audio_complete = threading.Event()

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

    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket connection closed: {close_status_code} - {close_msg}")

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

    def send_message(self, message):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.ws.send(json.dumps(message))
        else:
            raise ConnectionError("WebSocket is not connected")

    def close(self):
        if self.ws:
            self.ws.close()
        if self.websocket_thread:
            self.websocket_thread.join()

persistent_ws = PersistentWebSocket()
persistent_ws.connect()

def text_to_audio_array(text, prompt="Read the following text exactly as it is:"):
    persistent_ws.audio_string = ""
    persistent_ws.audio_complete.clear()

    try:
        persistent_ws.send_message({
            "type": "response.create",
            "response": {
                "modalities": ['audio', 'text'],
                "instructions": f"{prompt} {text}",
            }
        })

        # Wait for the audio transmission to complete
        persistent_ws.audio_complete.wait()

        if not persistent_ws.audio_string:
            raise ValueError("No audio received.")

        # Process the audio
        raw_pcm_data = base64.b64decode(persistent_ws.audio_string)
        audio_array = np.frombuffer(raw_pcm_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0

        # Resample to 16 kHz
        original_sample_rate = 24000  # Assuming the original sample rate is 24 kHz
        target_sample_rate = 16000
        resampled_audio = librosa.resample(audio_float, orig_sr=original_sample_rate, target_sr=target_sample_rate)

        return resampled_audio

    except Exception as e:
        print(f"Error in text_to_audio_array: {e}")
        return None

# Remember to close the connection when you're done using it
# persistent_ws.close()