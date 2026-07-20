"""Tests for the Qwen Conversation STT platform."""

from __future__ import annotations

import base64
import io
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from custom_components.qwen_conversation.stt_backend import (
    QwenASRBackend,
    STTError,
    create_backend,
    pcm_to_wav,
)
from homeassistant.components import stt
from homeassistant.core import HomeAssistant


def test_pcm_to_wav_writes_a_valid_container() -> None:
    """Home Assistant gives us headerless PCM; DashScope needs a container."""
    pcm = b"\x00\x01" * 1600

    wav = pcm_to_wav(pcm, 16000)

    assert wav.startswith(b"RIFF")
    with wave.open(io.BytesIO(wav), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        assert handle.readframes(handle.getnframes()) == pcm


def test_create_backend_rejects_unknown() -> None:
    """An unregistered provider id is an error."""
    with pytest.raises(STTError, match="Unknown STT backend"):
        create_backend("nope", MagicMock())


def _completion(text: str):
    """Build a minimal chat completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


async def test_transcribe_sends_base64_wav() -> None:
    """Audio is inlined as a data URI on the chat endpoint."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_completion("  turn on the lights  ")
    )
    backend = QwenASRBackend(client)

    pcm = b"\x00\x01" * 800
    text = await backend.async_transcribe(pcm, "qwen3-asr-flash", "en-US", 16000)

    assert text == "turn on the lights"

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "qwen3-asr-flash"
    # Only the leading subtag is meaningful to DashScope.
    assert kwargs["extra_body"]["asr_options"]["language"] == "en"

    part = kwargs["messages"][0]["content"][0]
    assert part["type"] == "input_audio"
    data_uri = part["input_audio"]["data"]
    assert data_uri.startswith("data:audio/wav;base64,")
    assert base64.b64decode(data_uri.split(",", 1)[1]) == pcm_to_wav(pcm, 16000)


async def test_transcribe_without_language_autodetects() -> None:
    """An unknown language is left to DashScope to detect."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_completion("hola"))

    await QwenASRBackend(client).async_transcribe(b"\x00\x01", "qwen3-asr-flash", "", 16000)

    asr_options = client.chat.completions.create.await_args.kwargs["extra_body"][
        "asr_options"
    ]
    assert "language" not in asr_options


async def test_transcribe_auth_error_is_actionable() -> None:
    """A rejected key produces a message telling the user to re-auth."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=openai.AuthenticationError(
            "bad key", response=MagicMock(status_code=401), body=None
        )
    )

    with pytest.raises(STTError, match="Re-authenticate"):
        await QwenASRBackend(client).async_transcribe(
            b"\x00\x01", "qwen3-asr-flash", "en-US", 16000
        )


async def test_transcribe_out_of_quota_does_not_retry() -> None:
    """Retrying an exhausted balance is pointless, so it fails immediately."""
    client = MagicMock()
    error = openai.RateLimitError(
        "no quota", response=MagicMock(status_code=429), body=None
    )
    error.code = "insufficient_quota"
    client.chat.completions.create = AsyncMock(side_effect=error)

    with pytest.raises(STTError, match="no remaining quota"):
        await QwenASRBackend(client).async_transcribe(
            b"\x00\x01", "qwen3-asr-flash", "en-US", 16000
        )

    assert client.chat.completions.create.await_count == 1


async def test_stt_entity_transcribes(hass: HomeAssistant, init_integration) -> None:
    """The entity buffers the stream and returns a successful result."""
    entity = stt.async_get_speech_to_text_entity(
        hass, "stt.qwen_conversation_speech_to_text"
    )
    assert entity is not None

    async def audio_stream():
        yield b"\x00\x01" * 400
        yield b"\x00\x01" * 400

    metadata = stt.SpeechMetadata(
        language="en-US",
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )

    init_integration.runtime_data.client.chat.completions.create = AsyncMock(
        return_value=_completion("hello there")
    )

    result = await entity.async_process_audio_stream(metadata, audio_stream())

    assert result.result == stt.SpeechResultState.SUCCESS
    assert result.text == "hello there"


async def test_stt_entity_rejects_empty_audio(
    hass: HomeAssistant, init_integration
) -> None:
    """An empty stream is an error rather than an empty transcript."""
    entity = stt.async_get_speech_to_text_entity(
        hass, "stt.qwen_conversation_speech_to_text"
    )

    async def empty_stream():
        return
        yield  # pragma: no cover

    metadata = stt.SpeechMetadata(
        language="en-US",
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )

    result = await entity.async_process_audio_stream(metadata, empty_stream())

    assert result.result == stt.SpeechResultState.ERROR
