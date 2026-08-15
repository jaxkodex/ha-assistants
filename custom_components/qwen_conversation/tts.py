"""Text-to-speech platform backed by Alibaba's speech models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from homeassistant.components.tts import (
    ATTR_AUDIO_OUTPUT,
    ATTR_VOICE,
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import QwenConfigEntry
from .const import (
    CONF_TTS_BACKEND,
    CONF_TTS_MODEL,
    CONF_TTS_VOICE,
    DEFAULT_TTS_BACKEND,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    TTS_BACKEND_QWEN,
)
from .entity import QwenBaseEntity
from .tts_backend import create_backend

# DashScope's speech models are multilingual and are not selected per language,
# so the entity accepts everything Assist is likely to send it.
SUPPORTED_LANGUAGES = [
    "zh-CN",
    "zh-HK",
    "zh-TW",
    "en-US",
    "en-GB",
    "ja-JP",
    "ko-KR",
    "es-ES",
    "fr-FR",
    "de-DE",
    "it-IT",
    "pt-PT",
    "ru-RU",
    "ar-SA",
    "id-ID",
    "th-TH",
    "vi-VN",
]


async def async_setup_entry(
    hass: HomeAssistant,
    entry: QwenConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the TTS entity."""
    async_add_entities([QwenTTSEntity(entry)])


class QwenTTSEntity(TextToSpeechEntity, QwenBaseEntity):
    """Speak text through the configured Alibaba speech backend."""

    _attr_supported_options = [ATTR_VOICE, ATTR_AUDIO_OUTPUT]
    _attr_supported_languages = SUPPORTED_LANGUAGES
    _attr_default_language = "en-US"
    # All three entities share one device, so only the conversation agent takes
    # the bare device name.
    _attr_name = "Text-to-speech"

    def __init__(self, entry: QwenConfigEntry) -> None:
        """Initialise the TTS entity."""
        QwenBaseEntity.__init__(self, entry)
        self._attr_unique_id = f"{entry.entry_id}-tts"

    @property
    def _backend_id(self) -> str:
        return self.entry.options.get(CONF_TTS_BACKEND, DEFAULT_TTS_BACKEND)

    @property
    def _model(self) -> str:
        return self.entry.options.get(CONF_TTS_MODEL, DEFAULT_TTS_MODEL)

    def _create_backend(self):
        """Build the backend for the current options.

        CosyVoice speaks websocket, Qwen-TTS speaks HTTP, so each gets the
        endpoint matching its protocol from the shared config entry.
        """
        runtime = self.entry.runtime_data
        backend_id = self._backend_id
        endpoint = (
            runtime.speech_http_url
            if backend_id == TTS_BACKEND_QWEN
            else runtime.speech_ws_url
        )
        return create_backend(backend_id, self.hass, runtime.speech_api_key, endpoint)

    @property
    def default_options(self) -> Mapping[str, Any]:
        """Return the configured voice as the default."""
        return {
            ATTR_VOICE: self.entry.options.get(CONF_TTS_VOICE, DEFAULT_TTS_VOICE),
            ATTR_AUDIO_OUTPUT: "mp3",
        }

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice] | None:
        """List the voices valid for the selected model."""
        voices = self._create_backend().voices_for_model(self._model)
        if not voices:
            return None
        return [Voice(voice_id=voice, name=voice) for voice in voices]

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Synthesise ``message`` and return the audio."""
        voice = options.get(ATTR_VOICE) or self.entry.options.get(
            CONF_TTS_VOICE, DEFAULT_TTS_VOICE
        )
        audio_format = options.get(ATTR_AUDIO_OUTPUT) or "mp3"

        return await self._create_backend().async_synthesize(
            message, self._model, voice, audio_format
        )
