"""Tests for the Qwen Conversation config entry setup."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import openai
import pytest

from custom_components.qwen_conversation import (
    resolve_endpoints,
    resolve_speech_endpoints,
)
from custom_components.qwen_conversation.const import (
    BASE_URLS,
    CONF_BASE_URL,
    CONF_REGION,
    CONF_SPEECH_API_KEY,
    CONF_SPEECH_BASE_URL,
    CONF_SPEECH_REGION,
    CONF_WS_URL,
    HTTP_URLS,
    REGION_CN,
    REGION_CUSTOM,
    REGION_INTL,
    REGION_SAME,
    WS_URLS,
)
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant


@pytest.mark.parametrize("region", [REGION_INTL, REGION_CN])
def test_resolve_endpoints_from_region(region: str) -> None:
    """A region preset resolves all three protocol endpoints."""
    base_url, ws_url, http_url = resolve_endpoints({CONF_REGION: region})

    assert base_url == BASE_URLS[region]
    assert ws_url == WS_URLS[region]
    assert http_url == HTTP_URLS[region]


def test_resolve_endpoints_derives_siblings_for_custom_host() -> None:
    """A workspace-scoped host gets matching websocket and HTTP URLs."""
    base_url, ws_url, http_url = resolve_endpoints(
        {
            CONF_REGION: REGION_CUSTOM,
            CONF_BASE_URL: "https://ws-1.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        }
    )

    assert base_url.endswith("/compatible-mode/v1")
    assert ws_url == "wss://ws-1.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/inference"
    assert http_url == "https://ws-1.ap-southeast-1.maas.aliyuncs.com/api/v1"


def test_resolve_endpoints_honours_explicit_ws_override() -> None:
    """An explicit websocket URL wins over the derived one."""
    _, ws_url, _ = resolve_endpoints(
        {
            CONF_REGION: REGION_CUSTOM,
            CONF_BASE_URL: "https://a.example/compatible-mode/v1",
            CONF_WS_URL: "wss://b.example/api-ws/v1/inference",
        }
    )

    assert ws_url == "wss://b.example/api-ws/v1/inference"


def test_speech_endpoints_default_to_the_conversation_ones() -> None:
    """Without a speech region, speech reuses the conversation endpoint."""
    data = {CONF_REGION: REGION_INTL, CONF_API_KEY: "chat-key"}

    base_url, ws_url, http_url, api_key = resolve_speech_endpoints(data)

    assert (base_url, ws_url, http_url) == resolve_endpoints(data)
    assert api_key == "chat-key"


def test_speech_endpoints_can_target_a_second_region() -> None:
    """Speech can live in another region than the chat models."""
    base_url, ws_url, http_url, api_key = resolve_speech_endpoints(
        {
            CONF_REGION: REGION_CUSTOM,
            CONF_BASE_URL: "https://eu.example/compatible-mode/v1",
            CONF_API_KEY: "chat-key",
            CONF_SPEECH_REGION: REGION_INTL,
            CONF_SPEECH_API_KEY: "speech-key",
        }
    )

    assert base_url == BASE_URLS[REGION_INTL]
    assert ws_url == WS_URLS[REGION_INTL]
    assert http_url == HTTP_URLS[REGION_INTL]
    assert api_key == "speech-key"


def test_speech_endpoints_fall_back_to_the_chat_key() -> None:
    """A shared key stays usable when only the region differs."""
    _, _, _, api_key = resolve_speech_endpoints(
        {
            CONF_REGION: REGION_CN,
            CONF_API_KEY: "chat-key",
            CONF_SPEECH_REGION: REGION_INTL,
        }
    )

    assert api_key == "chat-key"


def test_speech_endpoints_derive_siblings_for_a_custom_speech_host() -> None:
    """A custom speech host gets matching websocket and HTTP URLs."""
    _, ws_url, http_url, _ = resolve_speech_endpoints(
        {
            CONF_REGION: REGION_INTL,
            CONF_API_KEY: "chat-key",
            CONF_SPEECH_REGION: REGION_CUSTOM,
            CONF_SPEECH_BASE_URL: "https://sp.example/compatible-mode/v1",
        }
    )

    assert ws_url == "wss://sp.example/api-ws/v1/inference"
    assert http_url == "https://sp.example/api/v1"


def test_speech_region_same_is_the_documented_sentinel() -> None:
    """An explicit "same" behaves like an absent speech region."""
    data = {CONF_REGION: REGION_CN, CONF_API_KEY: "chat-key"}

    assert resolve_speech_endpoints({**data, CONF_SPEECH_REGION: REGION_SAME}) == (
        resolve_speech_endpoints(data)
    )


async def test_setup_and_unload(
    hass: HomeAssistant, mock_config_entry, mock_openai
) -> None:
    """The entry loads all three platforms and unloads cleanly."""
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.LOADED

    assert await hass.config_entries.async_unload(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.NOT_LOADED


async def test_setup_triggers_reauth_on_bad_key(
    hass: HomeAssistant, mock_config_entry, mock_openai
) -> None:
    """A rejected key puts the entry into the re-auth state."""
    mock_openai.with_options.return_value.models.list = AsyncMock(
        side_effect=openai.AuthenticationError(
            "bad key", response=AsyncMock(status_code=401), body=None
        )
    )

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.SETUP_ERROR


async def test_setup_retries_when_endpoint_unreachable(
    hass: HomeAssistant, mock_config_entry, mock_openai
) -> None:
    """A network failure is retried rather than treated as a bad key."""
    mock_openai.with_options.return_value.models.list = AsyncMock(
        side_effect=openai.APIConnectionError(request=AsyncMock())
    )

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_setup_tolerates_missing_models_endpoint(
    hass: HomeAssistant, mock_config_entry, mock_openai
) -> None:
    """Deployments without GET /models still load."""
    mock_openai.with_options.return_value.models.list = AsyncMock(
        side_effect=openai.NotFoundError(
            "no such route", response=AsyncMock(status_code=404), body=None
        )
    )

    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.LOADED
