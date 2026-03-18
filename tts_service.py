import os
from typing import Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def get_tts_client() -> boto3.client:
    """
    Create and return a boto3 Polly client.
    Credentials and region are taken from environment / usual AWS config chain.
    """
    region_name = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    return boto3.client("polly", region_name=region_name)


def synthesize_speech(
    text: str,
    voice_id: str = "Joanna",
    language_code: Optional[str] = "en-US",
    engine: str = "standard",
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Synthesize speech for the given text using AWS Polly.

    Returns (audio_bytes, error_message). On success, error_message is None.
    """
    if not text:
        return None, "No text provided for TTS."

    client = get_tts_client()

    try:
        params = {
            "OutputFormat": "mp3",
            "Text": text,
            "VoiceId": voice_id,
        }
        if language_code:
            params["LanguageCode"] = language_code
        if engine:
            params["Engine"] = engine

        response = client.synthesize_speech(**params)
        audio_stream = response.get("AudioStream")
        if audio_stream is None:
            return None, "No audio stream returned from TTS service."

        audio_bytes = audio_stream.read()
        return audio_bytes, None
    except (BotoCoreError, ClientError) as e:
        return None, f"TTS service error: {e}"

