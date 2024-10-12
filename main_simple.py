from text_to_audio_numpy import text_to_audio_array
from datasets import load_dataset, Audio
import multiprocessing
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_dataset_with_tts(dataset):
    def process_row(row):
        try:
            audio = text_to_audio_array(row['text'])
            row['audio'] = {
                'array': audio,
                'sampling_rate': 16000
            }
            return row
        except Exception as e:
            logger.error(f"Error processing row: {e}")
            return None  # Returning None will cause this row to be filtered out

    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    # Process the dataset using multithreading
    processed_dataset = dataset.map(
        process_row,
        num_proc=5,
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