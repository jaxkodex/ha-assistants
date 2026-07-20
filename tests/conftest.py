"""Fixtures for the Qwen Conversation tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.qwen_conversation.const import (
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_REGION,
    CONF_TTS_MODEL,
    CONF_TTS_VOICE,
    DOMAIN,
    REGION_INTL,
)
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Load the custom integration in every test."""
    return


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Return a config entry pointing at the Singapore endpoint."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Qwen Conversation",
        data={CONF_API_KEY: "sk-test", CONF_REGION: REGION_INTL},
        options={
            CONF_CHAT_MODEL: "qwen3.7-plus",
            CONF_TTS_MODEL: "cosyvoice-v3-flash",
            CONF_TTS_VOICE: "longxiaochun_v3",
        },
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def mock_openai():
    """Patch the OpenAI client so no network call is made during setup."""
    with patch(
        "custom_components.qwen_conversation.openai.AsyncOpenAI"
    ) as mock_client:
        client = mock_client.return_value
        client.platform_headers = lambda: {}
        client.with_options.return_value.models.list = AsyncMock(return_value=[])
        yield client


@pytest.fixture
async def init_integration(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry, mock_openai
) -> MockConfigEntry:
    """Set up the integration and return its entry."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    return mock_config_entry
