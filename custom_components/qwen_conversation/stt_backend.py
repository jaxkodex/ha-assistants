"""Pluggable speech-to-text backends.

Mirrors :mod:`tts_backend`: a backend turns raw audio into text, and backends
are resolved by id from :data:`BACKENDS` so a new provider is one class plus a
registry entry.

Only Alibaba ships today. :class:`QwenASRBackend` uses the OpenAI-compatible
chat endpoint — DashScope has no ``/audio/transcriptions`` route, but
``qwen3-asr-flash`` accepts an inline base64 data URI as an ``input_audio``
message part, which keeps the whole path async and reuses the client already
built for the conversation agent.
"""

from __future__ import annotations

import asyncio
import base64
import io
import wave
from abc import ABC, abstractmethod
from typing import Final

import openai

from homeassistant.exceptions import HomeAssistantError

from .const import (
    BACKOFF_MULTIPLIER,
    INITIAL_BACKOFF,
    LOGGER,
    MAX_RETRIES,
    QWEN_ASR_MODELS,
    STT_BACKEND_QWEN,
)


class STTError(HomeAssistantError):
    """Raised when a backend could not transcribe audio."""


class STTBackend(ABC):
    """A speech-to-text provider."""

    models: list[str] = []

    def __init__(self, client: openai.AsyncOpenAI) -> None:
        """Store the shared DashScope client."""
        self.client = client

    @abstractmethod
    async def async_transcribe(
        self, audio: bytes, model: str, language: str, sample_rate: int
    ) -> str:
        """Return the transcript of 16-bit mono PCM ``audio``."""


def pcm_to_wav(audio: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container.

    Home Assistant hands the stt platform headerless PCM, but DashScope needs a
    container it can identify from the data URI's MIME type.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)
    return buffer.getvalue()


class QwenASRBackend(STTBackend):
    """Qwen-ASR over DashScope's OpenAI-compatible chat endpoint."""

    models = list(QWEN_ASR_MODELS)

    async def async_transcribe(
        self, audio: bytes, model: str, language: str, sample_rate: int
    ) -> str:
        """Transcribe ``audio``, retrying when DashScope throttles us."""
        wav = pcm_to_wav(audio, sample_rate)
        data_uri = f"data:audio/wav;base64,{base64.b64encode(wav).decode()}"

        # DashScope wants a bare language subtag, Assist gives us a locale.
        asr_options: dict[str, object] = {"enable_itn": True}
        if language:
            asr_options["language"] = language.split("-")[0].lower()

        delay = INITIAL_BACKOFF
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                completion = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {"data": data_uri},
                                }
                            ],
                        }
                    ],
                    extra_body={"asr_options": asr_options},
                )
            except openai.AuthenticationError as err:
                raise STTError(
                    "DashScope rejected the API key. Re-authenticate the "
                    "Qwen Conversation integration."
                ) from err
            except openai.RateLimitError as err:
                if getattr(err, "code", None) == "insufficient_quota":
                    raise STTError(
                        "The DashScope account has no remaining quota."
                    ) from err
                last_error = err
                if attempt < MAX_RETRIES - 1:
                    LOGGER.debug("DashScope throttled ASR, retrying in %.1fs", delay)
                    await asyncio.sleep(delay)
                    delay *= BACKOFF_MULTIPLIER
                continue
            except openai.OpenAIError as err:
                raise STTError(f"Error talking to DashScope: {err}") from err

            if not completion.choices:
                raise STTError("Qwen-ASR returned no transcription")
            return (completion.choices[0].message.content or "").strip()

        raise STTError(
            f"DashScope kept throttling after {MAX_RETRIES} attempts: {last_error}"
        )


BACKENDS: Final[dict[str, type[STTBackend]]] = {
    STT_BACKEND_QWEN: QwenASRBackend,
}


def create_backend(backend_id: str, client: openai.AsyncOpenAI) -> STTBackend:
    """Instantiate the backend registered under ``backend_id``."""
    try:
        backend_cls = BACKENDS[backend_id]
    except KeyError:
        raise STTError(f"Unknown STT backend: {backend_id}") from None
    return backend_cls(client)
