"""Tests for the Qwen Conversation TTS platform."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.qwen_conversation.const import (
    CONF_TTS_BACKEND,
    CONF_TTS_MODEL,
    CONF_TTS_VOICE,
    TTS_BACKEND_QWEN,
)
from custom_components.qwen_conversation.tts_backend import (
    CosyVoiceBackend,
    QwenTTSBackend,
    TTSError,
    _find_audio_url,
    _is_rate_limit,
    create_backend,
)
from homeassistant.components import tts
from homeassistant.core import HomeAssistant


def test_create_backend_rejects_unknown() -> None:
    """An unregistered provider id is an error, not a silent fallback."""
    with pytest.raises(TTSError, match="Unknown TTS backend"):
        create_backend("nope", MagicMock(), "key", "url")


def test_cosyvoice_voices_are_model_specific() -> None:
    """Voices must not leak across CosyVoice generations."""
    backend = CosyVoiceBackend(MagicMock(), "key", "wss://example/api-ws/v1/inference")

    assert "longxiaochun_v3" in backend.voices_for_model("cosyvoice-v3-flash")
    assert "longxiaochun_v3" not in backend.voices_for_model("cosyvoice-v2")
    assert "longxiaochun_v2" in backend.voices_for_model("cosyvoice-v2")
    # Unknown/newer models fall back to a free-text voice field.
    assert backend.voices_for_model("cosyvoice-v3.5-plus") == []


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"output": {"audio": {"url": "https://x/a.wav"}}}, "https://x/a.wav"),
        ({"output": {"audio": {}}}, None),
        ({"output": {}}, None),
        ({}, None),
    ],
)
def test_find_audio_url(payload: dict, expected: str | None) -> None:
    """The Qwen-TTS response is parsed defensively."""
    assert _find_audio_url(payload) == expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("TaskFailed: Throttling.BurstRate", True),
        ("HTTP 429 too many requests", True),
        ("limit_requests exceeded", True),
        ("TaskFailed: invalid voice", False),
    ],
)
def test_is_rate_limit(message: str, expected: bool) -> None:
    """Throttling is recognised from the dashscope SDK's untyped errors."""
    assert _is_rate_limit(Exception(message)) is expected


async def test_cosyvoice_synthesize(hass: HomeAssistant) -> None:
    """The blocking SDK is driven from an executor and returns audio."""
    backend = CosyVoiceBackend(hass, "sk-test", "wss://example/api-ws/v1/inference")
    synthesizer = MagicMock()
    synthesizer.call.return_value = b"ID3audio"

    with (
        patch.dict(
            "sys.modules",
            {
                "dashscope": MagicMock(),
                "dashscope.audio.tts_v2": MagicMock(
                    SpeechSynthesizer=MagicMock(return_value=synthesizer)
                ),
            },
        ),
    ):
        extension, audio = await backend.async_synthesize(
            "hello", "cosyvoice-v3-flash", "longxiaochun_v3", "mp3"
        )

    assert extension == "mp3"
    assert audio == b"ID3audio"
    synthesizer.call.assert_called_once_with("hello")


async def test_cosyvoice_empty_audio_raises(hass: HomeAssistant) -> None:
    """A wrong-generation voice yields no audio and must surface clearly."""
    backend = CosyVoiceBackend(hass, "sk-test", "wss://example/api-ws/v1/inference")
    synthesizer = MagicMock()
    synthesizer.call.return_value = None

    with patch.dict(
        "sys.modules",
        {
            "dashscope": MagicMock(),
            "dashscope.audio.tts_v2": MagicMock(
                SpeechSynthesizer=MagicMock(return_value=synthesizer)
            ),
        },
    ):
        with pytest.raises(TTSError, match="voice belongs to this model version"):
            await backend.async_synthesize(
                "hello", "cosyvoice-v3-flash", "longxiaochun", "mp3"
            )


async def test_qwen_tts_downloads_returned_url(
    hass: HomeAssistant, aioclient_mock
) -> None:
    """Qwen-TTS answers with a link, so the audio needs a second fetch."""
    endpoint = "https://dashscope-intl.aliyuncs.com/api/v1"
    aioclient_mock.post(
        f"{endpoint}/services/aigc/multimodal-generation/generation",
        json={"output": {"audio": {"url": "https://example.com/a.wav"}}},
    )
    aioclient_mock.get("https://example.com/a.wav", content=b"RIFFwav")

    backend = QwenTTSBackend(hass, "sk-test", endpoint)
    extension, audio = await backend.async_synthesize(
        "hello", "qwen3-tts-flash", "Cherry", "mp3"
    )

    assert extension == "wav"
    assert audio == b"RIFFwav"


async def test_tts_entity_uses_configured_backend(
    hass: HomeAssistant, init_integration
) -> None:
    """Switching the provider option switches the endpoint protocol."""
    hass.config_entries.async_update_entry(
        init_integration,
        options={**init_integration.options, CONF_TTS_BACKEND: TTS_BACKEND_QWEN},
    )
    await hass.async_block_till_done()

    state = hass.states.get("tts.qwen_conversation_text_to_speech")
    assert state is not None
