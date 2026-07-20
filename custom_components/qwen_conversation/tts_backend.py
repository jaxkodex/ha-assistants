"""Pluggable text-to-speech backends.

Every backend implements :class:`TTSBackend`: given text, a voice and an audio
format, return ``(extension, audio_bytes)``. Backends are looked up by id in
:data:`BACKENDS`, so adding a provider means writing one class and registering
it — nothing else in the integration needs to change.

Two Alibaba backends ship today:

* :class:`CosyVoiceBackend` — DashScope's websocket protocol via the blocking
  ``dashscope`` SDK, run in an executor.
* :class:`QwenTTSBackend`   — the plain-HTTP multimodal-generation endpoint,
  driven with Home Assistant's shared aiohttp session.

Both take their endpoint from the config entry rather than hard-coding a host,
so workspace-scoped and self-hosted deployments work unchanged.
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Any, Final

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    BACKOFF_MULTIPLIER,
    INITIAL_BACKOFF,
    LOGGER,
    MAX_RETRIES,
    TTS_BACKEND_COSYVOICE,
    TTS_BACKEND_QWEN,
)

# The dashscope SDK keeps its API key and endpoint in module-level globals, so
# two config entries pointing at different regions would race each other. Every
# call that touches those globals holds this lock.
_DASHSCOPE_LOCK: Final = threading.Lock()

# Extension -> the dashscope AudioFormat member implementing it.
_COSYVOICE_FORMATS: Final[dict[str, str]] = {
    "mp3": "MP3_22050HZ_MONO_256KBPS",
    "wav": "WAV_22050HZ_MONO_16BIT",
    "pcm": "PCM_22050HZ_MONO_16BIT",
}


class TTSError(HomeAssistantError):
    """Raised when a backend could not synthesise speech."""


class TTSAuthError(TTSError):
    """Raised when the provider rejected our credentials."""


class TTSRateLimitError(TTSError):
    """Raised when the provider throttled us and retries were exhausted."""


class TTSBackend(ABC):
    """A text-to-speech provider."""

    #: Models this backend can be configured with.
    models: list[str] = []

    def __init__(self, hass: HomeAssistant, api_key: str, endpoint: str) -> None:
        """Store the shared credentials and the resolved endpoint."""
        self.hass = hass
        self.api_key = api_key
        self.endpoint = endpoint

    @abstractmethod
    async def async_synthesize(
        self, text: str, model: str, voice: str, audio_format: str
    ) -> tuple[str, bytes]:
        """Return ``(extension, audio_bytes)`` for ``text``."""

    @abstractmethod
    def voices_for_model(self, model: str) -> list[str]:
        """Return the voices known to work with ``model``.

        May be empty when the provider's voice list for that model is unknown;
        callers should treat the result as a suggestion, not a whitelist.
        """


class CosyVoiceBackend(TTSBackend):
    """CosyVoice over DashScope's websocket API, via the blocking SDK."""

    def __init__(self, hass: HomeAssistant, api_key: str, endpoint: str) -> None:
        """Initialise the backend."""
        super().__init__(hass, api_key, endpoint)
        from .const import COSYVOICE_MODELS

        self.models = list(COSYVOICE_MODELS)

    def voices_for_model(self, model: str) -> list[str]:
        """Return the CosyVoice voices matching ``model``'s generation."""
        from .const import COSYVOICE_VOICES

        return list(COSYVOICE_VOICES.get(model, []))

    async def async_synthesize(
        self, text: str, model: str, voice: str, audio_format: str
    ) -> tuple[str, bytes]:
        """Synthesise ``text``, retrying on throttling."""
        return await _async_with_backoff(
            lambda: self.hass.async_add_executor_job(
                self._synthesize, text, model, voice, audio_format
            )
        )

    def _synthesize(
        self, text: str, model: str, voice: str, audio_format: str
    ) -> tuple[str, bytes]:
        """Blocking synthesis. Runs in an executor thread, never the event loop."""
        import dashscope
        from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

        fmt_name = _COSYVOICE_FORMATS.get(audio_format, _COSYVOICE_FORMATS["mp3"])
        fmt = getattr(AudioFormat, fmt_name)

        with _DASHSCOPE_LOCK:
            # SpeechSynthesizer reads the API key at construction time, so both
            # globals must be set before we instantiate it.
            dashscope.api_key = self.api_key
            dashscope.base_websocket_api_url = self.endpoint
            synthesizer = SpeechSynthesizer(model=model, voice=voice, format=fmt)
            # With no callback, call() blocks and returns the complete audio.
            audio = synthesizer.call(text)

        if not audio:
            raise TTSError(
                f"CosyVoice returned no audio for model {model} with voice {voice}. "
                "CosyVoice voices are tied to a model generation — check that the "
                "voice belongs to this model version."
            )

        return audio_format, bytes(audio)


class QwenTTSBackend(TTSBackend):
    """Qwen-TTS over the plain-HTTP multimodal-generation endpoint."""

    def __init__(self, hass: HomeAssistant, api_key: str, endpoint: str) -> None:
        """Initialise the backend."""
        super().__init__(hass, api_key, endpoint)
        from .const import QWEN_TTS_MODELS

        self.models = list(QWEN_TTS_MODELS)

    def voices_for_model(self, model: str) -> list[str]:
        """Return the Qwen-TTS voices; they are shared across the family."""
        from .const import QWEN_TTS_VOICES

        return list(QWEN_TTS_VOICES)

    async def async_synthesize(
        self, text: str, model: str, voice: str, audio_format: str
    ) -> tuple[str, bytes]:
        """Synthesise ``text`` and download the resulting audio file."""
        return await _async_with_backoff(
            lambda: self._async_synthesize(text, model, voice)
        )

    async def _async_synthesize(
        self, text: str, model: str, voice: str
    ) -> tuple[str, bytes]:
        session = async_get_clientsession(self.hass)
        url = f"{self.endpoint.rstrip('/')}/services/aigc/multimodal-generation/generation"

        async with session.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model, "input": {"text": text, "voice": voice}},
        ) as response:
            if response.status in (401, 403):
                raise TTSAuthError(
                    f"DashScope rejected the API key: {await response.text()}"
                )
            if response.status == 429:
                raise TTSRateLimitError(await response.text())
            if response.status != 200:
                raise TTSError(
                    f"Qwen-TTS request failed ({response.status}): "
                    f"{await response.text()}"
                )
            payload = await response.json()

        # Qwen-TTS answers with a link to the rendered file rather than bytes.
        audio_url = _find_audio_url(payload)
        if audio_url is None:
            raise TTSError(f"Qwen-TTS response contained no audio URL: {payload}")

        async with session.get(audio_url) as audio_response:
            if audio_response.status != 200:
                raise TTSError(
                    f"Could not download Qwen-TTS audio ({audio_response.status})"
                )
            audio = await audio_response.read()

        # The endpoint renders WAV regardless of what the entity asked for.
        return "wav", audio


def _find_audio_url(payload: dict[str, Any]) -> str | None:
    """Pull the audio URL out of a Qwen-TTS response."""
    audio = payload.get("output", {}).get("audio")
    if isinstance(audio, dict):
        url = audio.get("url")
        return url if isinstance(url, str) and url else None
    return None


async def _async_with_backoff(factory: Any) -> tuple[str, bytes]:
    """Call ``factory()`` , retrying rate-limit failures with exponential backoff.

    DashScope applies a burst limit on top of the steady-state rate limit, so a
    throttled request is often served fine a moment later.
    """
    delay = INITIAL_BACKOFF
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            return await factory()
        except (TTSAuthError, TTSError) as err:
            if isinstance(err, TTSAuthError) or not _is_rate_limit(err):
                raise
            last_error = err
        except Exception as err:  # noqa: BLE001 - dashscope raises bare Exceptions
            if not _is_rate_limit(err):
                raise TTSError(str(err)) from err
            last_error = err

        if attempt < MAX_RETRIES - 1:
            LOGGER.debug("TTS throttled, retrying in %.1fs", delay)
            await asyncio.sleep(delay)
            delay *= BACKOFF_MULTIPLIER

    raise TTSRateLimitError(
        f"DashScope kept throttling after {MAX_RETRIES} attempts: {last_error}"
    )


def _is_rate_limit(err: Exception) -> bool:
    """Return whether ``err`` looks like DashScope throttling.

    The dashscope SDK has no dedicated rate-limit exception — websocket task
    failures surface as bare ``Exception("TaskFailed: ...")`` — so we match on
    the documented throttling codes in the message.
    """
    if isinstance(err, TTSRateLimitError):
        return True
    message = str(err).lower()
    return any(
        token in message
        for token in ("429", "throttling", "rate limit", "limit_requests", "limit_burst_rate")
    )


BACKENDS: Final[dict[str, type[TTSBackend]]] = {
    TTS_BACKEND_COSYVOICE: CosyVoiceBackend,
    TTS_BACKEND_QWEN: QwenTTSBackend,
}


def create_backend(
    backend_id: str, hass: HomeAssistant, api_key: str, endpoint: str
) -> TTSBackend:
    """Instantiate the backend registered under ``backend_id``."""
    try:
        backend_cls = BACKENDS[backend_id]
    except KeyError:
        raise TTSError(f"Unknown TTS backend: {backend_id}") from None
    return backend_cls(hass, api_key, endpoint)
