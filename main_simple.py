from text_to_audio_numpy import text_to_audio_array
from datasets import load_dataset, Audio, Dataset
from emotional_phrases.sad import synonyms, phrases
import random

batch_number = 2

selected_emotion = "sad"
dataset = Dataset.from_dict({
    "text": phrases
})




def process_dataset_with_tts(dataset):
    def process_row(row):
        try:


            #randomly select a synonym
            selected_emotion = random.choice(synonyms)

            prompt = f"Read the following text in a really {selected_emotion} voice:"

            audio = text_to_audio_array(row['text'], prompt)
            row['audio'] = {
                'array': audio,
                'sampling_rate': 16000
            }
            row["emotion"] = selected_emotion.lower()
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


dataset = dataset.select(range(0, 10))
processed_ds = process_dataset_with_tts(dataset)

# Push the processed dataset to the Hub
processed_ds.push_to_hub(f"amuvarma/sentencessaddebug")

print("Done processing dataset.")