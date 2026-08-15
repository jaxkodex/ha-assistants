"""The Qwen Conversation integration.

A single config entry holds the DashScope credentials and endpoint; it sets up
three platforms that share them: a conversation agent talking to Qwen through
DashScope's OpenAI-compatible endpoint, a text-to-speech entity talking to
CosyVoice or Qwen-TTS, and a speech-to-text entity talking to Qwen-ASR.
"""

from __future__ import annotations

from dataclasses import dataclass

import openai

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .const import (
    BASE_URLS,
    CONF_BASE_URL,
    CONF_REGION,
    CONF_SPEECH_API_KEY,
    CONF_SPEECH_BASE_URL,
    CONF_SPEECH_REGION,
    CONF_SPEECH_WS_URL,
    CONF_WS_URL,
    DEFAULT_REGION,
    DEFAULT_SPEECH_REGION,
    DOMAIN,
    HTTP_URLS,
    LOGGER,
    REGION_SAME,
    WS_URLS,
)

PLATFORMS: tuple[Platform, ...] = (
    Platform.CONVERSATION,
    Platform.STT,
    Platform.TTS,
)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


@dataclass(slots=True)
class QwenRuntimeData:
    """Everything the platforms need, resolved once at setup.

    The ``speech_*`` fields exist because the speech models are not always
    served from the same region as the chat models. They fall back to their
    conversation counterparts, so a single-region deployment sees no
    difference — and ``speech_client`` is then literally ``client``.
    """

    client: openai.AsyncOpenAI
    api_key: str
    base_url: str
    ws_url: str
    http_url: str
    speech_client: openai.AsyncOpenAI
    speech_api_key: str
    speech_base_url: str
    speech_ws_url: str
    speech_http_url: str


type QwenConfigEntry = ConfigEntry[QwenRuntimeData]


def resolve_endpoints(data: dict) -> tuple[str, str, str]:
    """Return the (compatible-mode, websocket, http) URLs for an entry.

    A custom region stores its own base URL; the websocket and plain-HTTP hosts
    are then derived from it so the TTS backends follow the same deployment as
    the conversation agent.
    """
    region = data.get(CONF_REGION, DEFAULT_REGION)

    if (base_url := data.get(CONF_BASE_URL)) is None:
        base_url = BASE_URLS[region]

    if (ws_url := data.get(CONF_WS_URL)) is None:
        ws_url = WS_URLS.get(region) or _swap_path(base_url, "wss", "api-ws/v1/inference")

    http_url = HTTP_URLS.get(region) or _swap_path(base_url, "https", "api/v1")

    return base_url, ws_url, http_url


def resolve_speech_endpoints(data: dict) -> tuple[str, str, str, str]:
    """Return the (compatible-mode, websocket, http, api key) used for speech.

    Falls back to the conversation endpoint and key whenever no separate speech
    region is configured, so entries created before this option existed keep
    working unchanged.
    """
    region = data.get(CONF_SPEECH_REGION, DEFAULT_SPEECH_REGION)

    if region == REGION_SAME:
        base_url, ws_url, http_url = resolve_endpoints(data)
        return base_url, ws_url, http_url, data[CONF_API_KEY]

    if (base_url := data.get(CONF_SPEECH_BASE_URL)) is None:
        base_url = BASE_URLS[region]

    if (ws_url := data.get(CONF_SPEECH_WS_URL)) is None:
        ws_url = WS_URLS.get(region) or _swap_path(
            base_url, "wss", "api-ws/v1/inference"
        )

    http_url = HTTP_URLS.get(region) or _swap_path(base_url, "https", "api/v1")

    # Keys are region-specific, but a workspace may share one across regions,
    # so an explicit speech key is optional.
    api_key = data.get(CONF_SPEECH_API_KEY) or data[CONF_API_KEY]

    return base_url, ws_url, http_url, api_key


def _swap_path(base_url: str, scheme: str, path: str) -> str:
    """Rebuild ``base_url`` with a different scheme and path.

    Used for custom/workspace-scoped hosts, where only the compatible-mode URL
    is known and the sibling endpoints live on the same host.
    """
    without_scheme = base_url.split("://", 1)[-1]
    host = without_scheme.split("/", 1)[0]
    return f"{scheme}://{host}/{path}"


async def async_setup_entry(hass: HomeAssistant, entry: QwenConfigEntry) -> bool:
    """Set up Qwen Conversation from a config entry."""
    base_url, ws_url, http_url = resolve_endpoints(dict(entry.data))
    api_key = entry.data[CONF_API_KEY]

    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=get_async_client(hass),
    )
    # Reading platform headers touches the filesystem and the platform module.
    await hass.async_add_executor_job(client.platform_headers)

    try:
        await client.with_options(timeout=10.0).models.list()
    except openai.AuthenticationError as err:
        raise ConfigEntryAuthFailed(
            f"Invalid DashScope API key: {err}"
        ) from err
    except openai.APIConnectionError as err:
        raise ConfigEntryNotReady(
            f"Could not reach DashScope at {base_url}: {err}"
        ) from err
    except openai.OpenAIError as err:
        # Workspace-scoped deployments do not always implement ``GET /models``.
        # That is not fatal — only auth and connectivity failures are.
        LOGGER.debug("Could not list models on %s, continuing anyway: %s", base_url, err)

    (
        speech_base_url,
        speech_ws_url,
        speech_http_url,
        speech_api_key,
    ) = resolve_speech_endpoints(dict(entry.data))

    # Only pay for a second client when the speech endpoint actually differs;
    # the common case is one region for everything.
    if (speech_base_url, speech_api_key) == (base_url, api_key):
        speech_client = client
    else:
        LOGGER.debug("Speech platforms using separate endpoint %s", speech_base_url)
        speech_client = openai.AsyncOpenAI(
            api_key=speech_api_key,
            base_url=speech_base_url,
            http_client=get_async_client(hass),
        )
        await hass.async_add_executor_job(speech_client.platform_headers)

    entry.runtime_data = QwenRuntimeData(
        client=client,
        api_key=api_key,
        base_url=base_url,
        ws_url=ws_url,
        http_url=http_url,
        speech_client=speech_client,
        speech_api_key=speech_api_key,
        speech_base_url=speech_base_url,
        speech_ws_url=speech_ws_url,
        speech_http_url=speech_http_url,
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_options))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: QwenConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_update_options(hass: HomeAssistant, entry: QwenConfigEntry) -> None:
    """Reload the entry when its options change."""
    await hass.config_entries.async_reload(entry.entry_id)
