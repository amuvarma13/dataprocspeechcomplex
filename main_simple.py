from text_to_audio_numpy import text_to_audio_array
from datasets import load_dataset, Audio, Dataset

import random

dsn = load_dataset("amuvarma/emo-texts-10")
dataset =  dsn['train']



def process_dataset_with_tts(dataset):
    def process_row(row):
        try:
            emotion = row["emotion"]
            prompt = f"Read the following text in a really {emotion} voice:"

            audio = text_to_audio_array(row['text'], prompt)
            row['audio'] = {
                'array': audio,
                'sampling_rate': 16000
            }
            row["emotion"] = emotion.lower()
            return row
        except Exception as e:
            print(f"Error processing row: {e}")
            return None  # Returning None will cause this row to be filtered out


    processed_dataset = dataset.map(
        process_row,
        num_proc=1,
    )
    
    processed_dataset = processed_dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    return processed_dataset


# dataset = dataset.select(range(0, 10))
processed_ds = process_dataset_with_tts(dataset)

# Push the processed dataset to the Hub
processed_ds.push_to_hub(f"amuvarma/sentencesangry100")

print("Done processing dataset.")