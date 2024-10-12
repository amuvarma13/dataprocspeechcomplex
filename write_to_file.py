import numpy as np
import wave
import struct

def write_numpy_to_wav(numpy_array, filename, sample_rate=16000, bit_depth=16):
  
    # Ensure the input is a numpy array
    numpy_array = np.asarray(numpy_array)

    # Normalize the array to the range [-1, 1] if it's not already
    if numpy_array.max() > 1 or numpy_array.min() < -1:
        numpy_array = numpy_array / np.max(np.abs(numpy_array))

    # Scale and convert to the appropriate integer type
    if bit_depth == 16:
        scaled = (numpy_array * 32767).astype(np.int16)
    elif bit_depth == 24:
        scaled = (numpy_array * 8388607).astype(np.int32)
    elif bit_depth == 32:
        scaled = (numpy_array * 2147483647).astype(np.int32)
    else:
        raise ValueError("Unsupported bit depth. Use 16, 24, or 32.")

    # Open the WAV file
    with wave.open(filename, 'wb') as wav_file:
        # Set the WAV file parameters
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)
        
        # Write the audio data
        wav_file.writeframes(scaled.tobytes())

    print(f"WAV file '{filename}' has been created successfully.")
