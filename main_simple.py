from text_to_audio_numpy import text_to_audio_array
from datasets import load_dataset, Audio
from emotion_list import all_emotions, common_emotions
import random

batch_number = 2



def process_dataset_with_tts(dataset):
    def process_row(row):
        try:
            #lets generate the prompt

            # generate 0 or 1 randomly
            prompt = None
            selected_emotion = None
            random_number = random.randint(0,1)

            if(random_number == 0):
                #pick a random element from common_emotions
                selected_emotion = random.choice(common_emotions)
                prompt = f"Read the following text in a {selected_emotion} tone:"
            else:
                #pick a random element from all_emotions
                selected_emotion = random.choice(all_emotions)
                prompt = f"Read the following text in a {selected_emotion} tone:"






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

# Load the dataset
ds = load_dataset(f"amuvarma/sentences1-audio")

# Process the dataset (assuming we're using the 'train' split)

# ds["train"] = ds["train"].select(range((batch_number-1)*200, 2*batch_number*200))
ds["train"] = ds["train"].select(range(0, 20))
processed_ds = process_dataset_with_tts(ds['train'])

# Push the processed dataset to the Hub
processed_ds.push_to_hub(f"amuvarma/sentencesdebug{batch_number}")

print("Done processing dataset.")