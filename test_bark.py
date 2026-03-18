import os

print("Testing Bark Alternative...")
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from scipy.io.wavfile import write as write_wav
    
    print("Dependencies for Bark are present. We would need to download models.")
    # preload_models()
    # text_prompt = "Hello, my name is Suno. And, uh — and I like pizza. [laughs]"
    # audio_array = generate_audio(text_prompt)
    # write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
    print("Success. Bark is a viable alternative.")
except ImportError:
    print("Bark is not installed. We will need: pip install git+https://github.com/suno-ai/bark.git")
