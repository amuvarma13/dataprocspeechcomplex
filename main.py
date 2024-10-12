from text_to_audio_numpy import text_to_audio_array
from datasets import load_dataset
import numpy as np

def process_dataset_with_tts(dataset):

    def process_row(row):
        all_audio = []
        for text in row['texts']:
            audio = text_to_audio_array(text)
            all_audio.append(audio)
        
        combined_audio = np.concatenate(all_audio)
        
        row['audio'] = {
            'array': combined_audio,
            'sampling_rate': 16000
        }
        return row

    return dataset.map(process_row)


ds = load_dataset("amuvarma/sentences1-3")


processed_ds = process_dataset_with_tts(ds['train'])  # Assuming we're using the 'train' split

processed_ds.push_to_hub("amuvarma/sentences1-3-audio")  # Push the processed dataset to the Hub

# Print info about the processed dataset
print(processed_ds)
print(f"Number of rows: {len(processed_ds)}")
print(f"Columns: {processed_ds.column_names}")

# Example: Print the length of the audio array for the first row
first_row_audio = processed_ds[0]['audio']['array']
print(f"Length of audio array in first row: {len(first_row_audio)}")