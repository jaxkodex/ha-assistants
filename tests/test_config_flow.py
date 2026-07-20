"""Tests for the Qwen Conversation config flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import openai
import pytest

from custom_components.qwen_conversation.const import (
    BASE_URLS,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_REGION,
    DOMAIN,
    REGION_CN,
    REGION_CUSTOM,
    REGION_INTL,
)
from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType


@pytest.fixture
def mock_validate():
    """Patch credential validation to succeed."""
    with patch(
        "custom_components.qwen_conversation.config_flow.validate_credentials",
        new=AsyncMock(return_value=None),
    ) as mock:
        yield mock


async def test_user_flow_creates_entry(
    hass: HomeAssistant, mock_validate
) -> None:
    """A region preset stores the matching base URL and the chosen model."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_API_KEY: "sk-test",
            CONF_REGION: REGION_CN,
            CONF_CHAT_MODEL: "qwen3.7-plus",
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_API_KEY: "sk-test", CONF_REGION: REGION_CN}
    # The model lives in options so it can be changed without re-auth.
    assert result["options"][CONF_CHAT_MODEL] == "qwen3.7-plus"
    mock_validate.assert_awaited_once()
    assert mock_validate.await_args.args[2] == BASE_URLS[REGION_CN]


async def test_custom_region_asks_for_endpoint(
    hass: HomeAssistant, mock_validate
) -> None:
    """Choosing Custom collects a workspace-scoped base URL."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_API_KEY: "sk-test",
            CONF_REGION: REGION_CUSTOM,
            CONF_CHAT_MODEL: "qwen3.7-plus",
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "endpoint"

    custom_url = "https://ws-123.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_BASE_URL: custom_url}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_BASE_URL] == custom_url
    assert mock_validate.await_args.args[2] == custom_url


@pytest.mark.parametrize(
    ("side_effect", "expected_error"),
    [
        (
            openai.AuthenticationError(
                "bad key", response=AsyncMock(status_code=401), body=None
            ),
            "invalid_auth",
        ),
        (openai.APIConnectionError(request=AsyncMock()), "cannot_connect"),
        (RuntimeError("boom"), "unknown"),
    ],
)
async def test_user_flow_errors(
    hass: HomeAssistant, side_effect: Exception, expected_error: str
) -> None:
    """Validation failures are reported on the form, not raised."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )

    with patch(
        "custom_components.qwen_conversation.config_flow.validate_credentials",
        new=AsyncMock(side_effect=side_effect),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_API_KEY: "bad",
                CONF_REGION: REGION_INTL,
                CONF_CHAT_MODEL: "qwen3.7-plus",
            },
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": expected_error}


async def test_reauth_updates_key(
    hass: HomeAssistant, mock_config_entry, mock_validate
) -> None:
    """Re-auth replaces the key and keeps the configured endpoint."""
    result = await mock_config_entry.start_reauth_flow(hass)
    assert result["step_id"] == "reauth_confirm"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-new"}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert mock_config_entry.data[CONF_API_KEY] == "sk-new"
    assert mock_config_entry.data[CONF_REGION] == REGION_INTL
