import websocket
import json
import base64
import numpy as np
import librosa
import threading
import os

url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
apikey = os.getenv("OPENAI_API_KEY")

def text_to_audio_array(text, prompt="Read the following text exactly as it is:"):
  
    audio_string = ""
    audio_complete = threading.Event()

    def on_message(ws, message):
        nonlocal audio_string
        socket_message = json.loads(message)
        message_type = socket_message["type"]

        if message_type == "response.audio.delta":
            delta = socket_message["delta"]
            audio_string += delta
        elif message_type == "response.done":
            audio_complete.set()

    def on_open(ws):

        ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ['audio', 'text'],
                "instructions": f"{prompt} {text}",
            }
        }))

    ws = websocket.WebSocketApp(
        url,
        on_message=on_message,
        on_open=on_open,
        header={
            "Authorization": f"Bearer {apikey}",
            "OpenAI-Beta": "realtime=v1",
        }
    )

    websocket_thread = threading.Thread(target=ws.run_forever)
    websocket_thread.start()

    # Wait for the audio transmission to complete
    audio_complete.wait()

    # Close the WebSocket connection
    ws.close()
    websocket_thread.join()

    if not audio_string:
        raise ValueError("No audio received.")

    # Process the audio
    raw_pcm_data = base64.b64decode(audio_string)
    audio_array = np.frombuffer(raw_pcm_data, dtype=np.int16)
    audio_float = audio_array.astype(np.float32) / 32768.0

    # Resample to 16 kHz
    original_sample_rate = 24000  # Assuming the original sample rate is 24 kHz
    target_sample_rate = 16000
    resampled_audio = librosa.resample(audio_float, orig_sr=original_sample_rate, target_sr=target_sample_rate)

    return resampled_audio
