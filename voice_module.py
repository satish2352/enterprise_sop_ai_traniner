import os
import io
import scipy.io.wavfile
import warnings
import torch

# Suppress warnings that bark sometimes throws
warnings.filterwarnings("ignore")

# Define where to save 
AUDIO_DIR = "./audio_outputs"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Force Bark to use CUDA if available for the RTX 4060
os.environ["SUNO_USE_SMALL_MODELS"] = "True" # Use small models for speed on local GPUs

def generate_voice(text_prompt: str, filename: str = "output.wav") -> str:
    """
    Generates audio from text using Suno Bark and saves it to a wav file.
    Returns the file path of the generated audio.
    """
    try:
        from bark import SAMPLE_RATE, generate_audio, preload_models
        
        # PyTorch 2.6+ defaults to weights_only=True which blocks Bark models. Overriding globally.
        _original_load = torch.load
        def _legacy_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)
        torch.load = _legacy_load
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Bark models on {device}...")
        
        # We try to preload, but if it's already in memory Bark handles it
        preload_models()
        
        print(f"Generating audio for text: {text_prompt[:50]}...")
        # Generate audio array
        # We don't use history_prompt to reduce latency, or you can supply it if you want specific voice clone
        audio_array = generate_audio(text_prompt, history_prompt="v2/en_speaker_6") 
        
        output_path = os.path.join(AUDIO_DIR, filename)
        
        # Save to wav
        scipy.io.wavfile.write(output_path, rate=SAMPLE_RATE, data=audio_array)
        
        print(f"Audio saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error generating voice via Bark: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    generate_voice("Hello, this is a test of the accelerated Bark text to speech system.")
