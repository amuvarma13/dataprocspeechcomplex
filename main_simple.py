from text_to_audio_numpy import text_to_audio_array, persistent_ws
from datasets import load_dataset, Audio
from time import sleep
import signal
import threading

def process_dataset_with_tts(dataset):
  
    def process_row_with_timeout(row, idx):
        print(f"Processing row {idx}...")
        
        def process():
            nonlocal audio
            try:
                audio = text_to_audio_array(row['text'])
            except Exception as e:
                print(f"Error processing row: {e}")

        audio = None
        thread = threading.Thread(target=process)
        thread.start()
        thread.join(timeout=10)  # Wait for up to 10 seconds

        if thread.is_alive():
            print(f"Row {idx} took too long to process. Setting empty audio array.")
            persistent_ws.reset_socket()
            sleep(10)
            audio = []  # Set to empty list if processing takes too long
        elif audio is None:
            print(f"Row {idx} failed to process. Setting empty audio array.")
            audio = []  # Set to empty list if processing fails

        row['audio'] = {
            'array': audio,
            'sampling_rate': 16000
        }
        return row
 
    # Process the dataset
    processed_dataset = dataset.map(
        process_row_with_timeout,
        num_proc=1,
        with_indices=True,
    )
    print("about to push")
    processed_dataset.push_to_hub("amuvarma/sentences1-audio-debug-0")   
    print("pushed")
    # Cast the 'audio' column to Audio feature
    processed_dataset = processed_dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    return processed_dataset
# Load the dataset
ds = load_dataset("amuvarma/sentences1")

# Process the dataset (assuming we're using the 'train' split)
ds["train"] = ds["train"].select(range(0,20))
processed_ds = process_dataset_with_tts(ds['train'])

# Push the processed dataset to the Hub
processed_ds.push_to_hub("amuvarma/sentences1-audio-3")

print("Done processing dataset.")

# Close the WebSocket connection when done
persistent_ws.close()