from text_to_audio_numpy import text_to_audio_array, persistent_ws
from datasets import load_dataset, Audio
import multiprocessing
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_dataset_with_tts(dataset):
    def process_row(row, idx):
        try:
            # Assuming the text is the first (and only) item in the list
            text = row[0]
            audio = text_to_audio_array(text)
            
            # Reset socket and sleep every 10 rows
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1} rows. Resetting socket...")
                persistent_ws.reset_socket()
                time.sleep(20)  # Wait for 20 seconds after reset
            
            return {
                'text': text,
                'audio': {
                    'array': audio,
                    'sampling_rate': 16000
                }
            }
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            return None  # Returning None will cause this row to be filtered out

    # Process the dataset
    processed_dataset = dataset.map(
        process_row,
        with_indices=True,
        num_proc=1,
    )
    
    # Filter out None values (failed rows)
    processed_dataset = processed_dataset.filter(lambda x: x is not None)
    
    # Cast the 'audio' column to Audio feature
    processed_dataset = processed_dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    return processed_dataset

# Load the dataset
ds = load_dataset("amuvarma/sentences1")

# Process the dataset (assuming we're using the 'train' split)
ds["train"] = ds["train"].select(range(200))
processed_ds = process_dataset_with_tts(ds['train'])

# Push the processed dataset to the Hub
processed_ds.push_to_hub("amuvarma/sentences1-audio-0-200")

print("Done processing dataset.")

# Close the WebSocket connection when done
persistent_ws.close()