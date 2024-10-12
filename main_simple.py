from text_to_audio_numpy import text_to_audio_array
from datasets import load_dataset, Audio, Dataset
from emotion_list import all_emotions, common_emotions
from emotional_phrases.happy import happy_phrases
import random

batch_number = 2

selected_emotion = "happy"
dataset = Dataset.from_dict({
    "text": happy_phrases
})




def process_dataset_with_tts(dataset):
    def process_row(row):
        try:


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


smalldataset = dataset.select(range(0, 20))
processed_ds = process_dataset_with_tts(smalldataset)

# Push the processed dataset to the Hub
processed_ds.push_to_hub(f"amuvarma/sentencesdebug{batch_number}")

print("Done processing dataset.")