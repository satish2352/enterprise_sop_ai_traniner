import os
import asyncio
import edge_tts
import traceback

AUDIO_DIR = "./audio_outputs"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

async def _generate_voice_async(text_prompt: str, output_path: str):
    # 'en-US-AriaNeural' is a great versatile female voice.
    # 'en-US-ChristopherNeural' is a good male voice.
    voice = "en-US-AriaNeural"
    communicate = edge_tts.Communicate(text_prompt, voice)
    await communicate.save(output_path)

def generate_voice(text_prompt: str, filename: str = "output.mp3") -> str:
    """
    Generates audio from text using Microsoft Edge TTS (ultra-fast) and saves it.
    Returns the file path of the generated audio.
    """
    try:
        # Edge TTS generates MP3s by default, change extension if needed
        if filename.endswith(".wav"):
            filename = filename.replace(".wav", ".mp3")
            
        output_path = os.path.join(AUDIO_DIR, filename)
        
        print(f"Generating audio for text (Edge TTS): {text_prompt[:50]}...")
        asyncio.run(_generate_voice_async(text_prompt, output_path))
        
        print(f"Audio saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error generating voice via Edge TTS: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import time
    t0 = time.time()
    generate_voice("Hello, this is a test of the ultra-fast Edge text to speech system.")
    print(f"Time taken: {time.time() - t0} seconds")
