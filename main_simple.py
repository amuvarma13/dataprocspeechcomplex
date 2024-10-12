from text_to_audio_numpy import text_to_audio_array, persistent_ws
from datasets import load_dataset, Audio
from time import sleep

def process_dataset_with_tts(dataset):
    def process_row(row, idx):
        print(f"Processing row {idx}...")
        if idx % 10 == 0:
            persistent_ws.reset_socket()
            sleep(10)

        try:
            audio = text_to_audio_array(row['text'])
            row['audio'] = {
                'array': audio,
                'sampling_rate': 16000
            }
            return row
        except Exception as e:
            print(f"Error processing row: {e}")
            return None  # Returning None will cause this row to be filtered out

    # Get the number of available CPU cores

    # Process the dataset using multithreading
    processed_dataset = dataset.map(
        process_row,
        num_proc=1,
        with_indices=True,
        remove_columns=dataset.column_names  # Remove original columns
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
processed_ds.push_to_hub("amuvarma/sentences1-audio")

print("Done processing dataset.")