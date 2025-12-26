from piper.voice import PiperVoice
import sounddevice as sd
import numpy as np

# Load the model
model_path = "en_US-lessac-medium.onnx"
voice = PiperVoice.load(model_path)

text = "i don't even care about you"

# Collect audio chunks
audio_chunks = []
for audio_chunk in voice.synthesize(text):
    # audio_chunk.audio_float_array is already a numpy float32 array
    audio_chunks.append(audio_chunk.audio_float_array)

# Combine all chunks into one array
audio_data = np.concatenate(audio_chunks)

# Get sample rate from the first chunk (or from voice.config)
sample_rate = voice.config.sample_rate

# Play immediately
print(f"Playing audio now at {sample_rate} Hz...")
sd.play(audio_data, samplerate=sample_rate)
sd.wait()
print("Done!")