"""Speech-to-text platform backed by Alibaba's ASR models."""

from __future__ import annotations

from collections.abc import AsyncIterable

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import QwenConfigEntry
from .const import (
    CONF_STT_BACKEND,
    CONF_STT_MODEL,
    DEFAULT_STT_BACKEND,
    DEFAULT_STT_MODEL,
    LOGGER,
    STT_LANGUAGES,
)
from .entity import QwenBaseEntity
from .stt_backend import STTError, create_backend

# Qwen-ASR caps a single request at 5 minutes / 10 MB. A voice command never
# comes close, but a stuck stream would otherwise buffer without bound.
MAX_AUDIO_BYTES = 10 * 1024 * 1024


async def async_setup_entry(
    hass: HomeAssistant,
    entry: QwenConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the STT entity."""
    async_add_entities([QwenSTTEntity(entry)])


class QwenSTTEntity(SpeechToTextEntity, QwenBaseEntity):
    """Transcribe Assist audio through the configured Alibaba ASR backend."""

    _attr_name = "Speech-to-text"

    def __init__(self, entry: QwenConfigEntry) -> None:
        """Initialise the STT entity."""
        QwenBaseEntity.__init__(self, entry)
        self._attr_unique_id = f"{entry.entry_id}-stt"

    @property
    def supported_languages(self) -> list[str]:
        """Return the languages Qwen-ASR can transcribe."""
        return STT_LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Assist delivers headerless PCM, which we wrap as WAV ourselves."""
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return supported codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return supported bit rates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return supported sample rates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return supported channel counts."""
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Buffer the incoming audio and transcribe it in one request."""
        audio = bytearray()
        async for chunk in stream:
            audio.extend(chunk)
            if len(audio) > MAX_AUDIO_BYTES:
                LOGGER.error("Audio stream exceeded %d bytes", MAX_AUDIO_BYTES)
                return SpeechResult(None, SpeechResultState.ERROR)

        if not audio:
            return SpeechResult(None, SpeechResultState.ERROR)

        options = self.entry.options
        backend = create_backend(
            options.get(CONF_STT_BACKEND, DEFAULT_STT_BACKEND),
            self.entry.runtime_data.client,
        )

        try:
            text = await backend.async_transcribe(
                bytes(audio),
                options.get(CONF_STT_MODEL, DEFAULT_STT_MODEL),
                metadata.language,
                int(metadata.sample_rate),
            )
        except STTError as err:
            LOGGER.error("Speech recognition failed: %s", err)
            return SpeechResult(None, SpeechResultState.ERROR)

        return SpeechResult(text, SpeechResultState.SUCCESS)
