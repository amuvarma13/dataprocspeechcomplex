from text_to_audio_numpy import text_to_audio_array
from datasets import load_dataset, Audio
import multiprocessing

def process_dataset_with_tts(dataset):
    def process_batch(batch):
        processed_batch = {
            'text': batch['text'],
            'audio': []
        }
        for text in batch['text']:
            audio = text_to_audio_array(text)
            print("audio received")
            processed_batch['audio'].append({
                'array': audio,
                'sampling_rate': 16000
            })
            print("audio appended")
        return processed_batch

    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    # Process the dataset using multithreading
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        num_proc=5,
    )
    
    # Cast the 'audio' column to Audio feature
    processed_dataset = processed_dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    return processed_dataset

# Load the dataset
ds = load_dataset("amuvarma/sentences1")

# Process the dataset (assuming we're using the 'train' split)
processed_ds = process_dataset_with_tts(ds['train'])

# Push the processed dataset to the Hub
processed_ds.push_to_hub("amuvarma/sentences1-audio")

# Print info about the processed dataset
print(processed_ds)
print(f"Number of rows: {len(processed_ds)}")
print(f"Columns: {processed_ds.column_names}")

# Example: Print the length of the audio array for the first row
first_row_audio = processed_ds[0]['audio']['array']
print(f"Length of audio array in first row: {len(first_row_audio)}")