from text_to_audio_numpy import text_to_audio_array, persistent_ws
from datasets import load_dataset, Audio
import numpy as np
import time

def process_dataset_with_tts(dataset):
    def process_row(row):
        all_audio = []
        for text in row['texts']:
            audio = text_to_audio_array(text)
            all_audio.append(audio)
        combined_audio = np.concatenate(all_audio)
        return {
            'texts': row['texts'],
            'audio': {
                'array': combined_audio,
                'sampling_rate': 16000
            }
        }

    processed_dataset = dataset.map(
        process_row,
        remove_columns=dataset.column_names
    )
    
    # Cast the 'audio' column to Audio feature
    processed_dataset = processed_dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    return processed_dataset

# Load the dataset
ds = load_dataset("amuvarma/sentences1-3")

# Process the dataset (assuming we're using the 'train' split)
processed_ds = ds['train'].map(
    lambda row, idx: {
        **process_dataset_with_tts(load_dataset("amuvarma/sentences1-3", split=f"train[{idx}:{idx+1}]"))['audio'],
        'texts': row['texts']
    },
    with_indices=True,
    remove_columns=ds['train'].column_names
)

# Reset socket every 10 rows
for i in range(0, len(processed_ds), 10):
    if i > 0:
        print(f"Processed {i} rows. Resetting socket...")
        persistent_ws.reset_socket()
        time.sleep(20)  # Wait for 20 seconds after reset
    
    # Process next 10 rows
    end = min(i + 10, len(processed_ds))
    processed_ds = processed_ds.add_item(processed_ds[i:end])

# Push the processed dataset to the Hub
processed_ds.push_to_hub("amuvarma/sentences1-3-audio")

# Close the WebSocket connection when done
persistent_ws.close()