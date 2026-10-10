"""Audio transcription service — uses OpenAI Whisper API via OpenAI to transcribe audio."""

import logging
import tempfile
from pathlib import Path

from app.services.openai_client import openai_client

logger = logging.getLogger(__name__)


def transcribe_audio(file_bytes: bytes, filename: str) -> str:
    """Transcribe audio using OpenAI Whisper API via OpenAI.

    Falls back to empty string if transcription fails.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / filename
        audio_path.write_bytes(file_bytes)

        try:
            with open(audio_path, "rb") as f:
                response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                )
            transcript = response.text or ""
            logger.info("Transcribed %s: %s", filename, transcript[:100])
            return transcript
        except Exception:
            logger.exception("Whisper transcription failed for %s, falling back to metadata", filename)
            return ""
