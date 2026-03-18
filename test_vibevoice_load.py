import torch
from transformers import AutoModelForCausalLM

print("Loading VibeVoice model...")
try:
    # As per VibeVoice docs and standard HuggingFace usage, testing if we can load the model
    # Note: VibeVoice typically requires specific code from their repo, but we'll try to load the 
    # model weights if it's hosted as a standard HF model.
    model = AutoModelForCausalLM.from_pretrained("microsoft/VibeVoice-1.5B", trust_remote_code=True)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load directly via transformers: {e}")
    print("VibeVoice will likely need to be initialized via its specific repository classes.")
