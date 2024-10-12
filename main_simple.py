from text_to_audio_numpy import text_to_audio_array
from datasets import load_dataset, Audio, Dataset
print("Libraries loaded.")



start = 3500
end = 4500
num_threads = 10

dsn = load_dataset("amuvarma/emotions-text-2")
dataset =  dsn['train']

print("Dataset loaded.")


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
        num_proc=num_threads,
    )
    
    processed_dataset = processed_dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    return processed_dataset


dataset = dataset.select(range(start, end))
processed_ds = process_dataset_with_tts(dataset)



# Push the processed dataset to the Hub
processed_ds.push_to_hub(f"amuvarma/emotions-2-{start}-{end}")

print("Done processing dataset.")