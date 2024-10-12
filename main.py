from text_to_audio_numpy import text_to_audio_array
from datasets import load_dataset, Audio
import numpy as np
import multiprocessing

def process_dataset_with_tts(dataset):
    def process_batch(batch):
        processed_batch = {
            'texts': batch['texts'],
            'audio': []
        }
        for texts in batch['texts']:
            all_audio = []
            for text in texts:
                audio = text_to_audio_array(text)
                all_audio.append(audio)
            combined_audio = np.concatenate(all_audio)
            processed_batch['audio'].append({
                'array': combined_audio,
                'sampling_rate': 16000
            })
        return processed_batch

    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    # Process the dataset using multithreading
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        num_proc=num_cores,
        remove_columns=dataset.column_names
    )
    
    # Cast the 'audio' column to Audio feature
    processed_dataset = processed_dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    return processed_dataset

# Load the dataset
ds = load_dataset("amuvarma/sentences1-3")

# Process the dataset (assuming we're using the 'train' split)
processed_ds = process_dataset_with_tts(ds['train'])

# Push the processed dataset to the Hub
processed_ds.push_to_hub("amuvarma/sentences1-3-audio")

# Print info about the processed dataset
print(processed_ds)
print(f"Number of rows: {len(processed_ds)}")
print(f"Columns: {processed_ds.column_names}")

# Example: Print the length of the audio array for the first row
first_row_audio = processed_ds[0]['audio']['array']
print(f"Length of audio array in first row: {len(first_row_audio)}")