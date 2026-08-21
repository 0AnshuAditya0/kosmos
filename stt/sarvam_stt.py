import os
import io

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()


class SarvamSTT:
    def __init__(self):
        api_key = os.getenv("SARVAM_API_KEY")

        if not api_key:
            raise RuntimeError("SARVAM_API_KEY is not set")

        self.client = SarvamAI(
            api_subscription_key=api_key
        )

    def transcribe(self, audio_bytes: bytes) -> dict:
        if not audio_bytes:
            raise ValueError("audio_bytes cannot be empty")

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"

        response = self.client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            mode="transcribe",
            language_code="unknown",
        )

        return {
            "text": response.transcript,
            "language": response.language_code,
        }


_default_stt = None


def transcribe(audio_bytes: bytes) -> dict:
    global _default_stt

    if _default_stt is None:
        _default_stt = SarvamSTT()

    return _default_stt.transcribe(audio_bytes)