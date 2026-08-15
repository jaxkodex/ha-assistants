"""Config flow for Qwen Conversation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import openai
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import callback
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from . import QwenConfigEntry, resolve_endpoints, resolve_speech_endpoints
from .stt_backend import BACKENDS as STT_BACKENDS
from .const import (
    BASE_URLS,
    CHAT_MODELS,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_ENABLE_THINKING,
    CONF_FOLLOW_UP,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REGION,
    CONF_SPEECH_API_KEY,
    CONF_SPEECH_BASE_URL,
    CONF_SPEECH_REGION,
    CONF_SPEECH_WS_URL,
    CONF_STT_BACKEND,
    CONF_STT_MODEL,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_TTS_BACKEND,
    CONF_TTS_MODEL,
    CONF_TTS_VOICE,
    CONF_WS_URL,
    COSYVOICE_MODELS,
    COSYVOICE_VOICES,
    DEFAULT_CHAT_MODEL,
    DEFAULT_ENABLE_THINKING,
    DEFAULT_FOLLOW_UP,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NAME,
    DEFAULT_REGION,
    DEFAULT_SPEECH_REGION,
    DEFAULT_STT_BACKEND,
    DEFAULT_STT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TTS_BACKEND,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    DOMAIN,
    LOGGER,
    QWEN_ASR_MODELS,
    QWEN_TTS_MODELS,
    QWEN_TTS_VOICES,
    REGION_CUSTOM,
    REGION_SAME,
    REGIONS,
    SPEECH_REGIONS,
    TTS_BACKEND_COSYVOICE,
    TTS_BACKEND_QWEN,
)


def _combobox(options: list[str]) -> SelectSelector:
    """A dropdown that also accepts a hand-typed value.

    Model IDs turn over quickly and dated snapshots such as
    ``qwen3.6-flash-2026-04-16`` are never in our list, so every model and
    voice field stays free-text.
    """
    return SelectSelector(
        SelectSelectorConfig(
            options=[SelectOptionDict(value=option, label=option) for option in options],
            mode=SelectSelectorMode.DROPDOWN,
            custom_value=True,
            sort=False,
        )
    )


STEP_USER_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
        vol.Required(CONF_REGION, default=DEFAULT_REGION): SelectSelector(
            SelectSelectorConfig(
                options=REGIONS,
                mode=SelectSelectorMode.DROPDOWN,
                translation_key="region",
            )
        ),
        vol.Required(CONF_CHAT_MODEL, default=DEFAULT_CHAT_MODEL): _combobox(
            CHAT_MODELS
        ),
    }
)

STEP_ENDPOINT_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_BASE_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Optional(CONF_WS_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
    }
)


STEP_SPEECH_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_SPEECH_REGION, default=DEFAULT_SPEECH_REGION): SelectSelector(
            SelectSelectorConfig(
                options=SPEECH_REGIONS,
                mode=SelectSelectorMode.DROPDOWN,
                translation_key="speech_region",
            )
        ),
        vol.Optional(CONF_SPEECH_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
        vol.Optional(CONF_SPEECH_BASE_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Optional(CONF_SPEECH_WS_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
    }
)


async def validate_credentials(hass, api_key: str, base_url: str) -> None:
    """Raise if DashScope will not accept these credentials.

    Not every deployment implements ``GET /models``; only authentication and
    connectivity problems are treated as failures.
    """
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=get_async_client(hass),
    )
    await hass.async_add_executor_job(client.platform_headers)
    try:
        await client.with_options(timeout=10.0).models.list()
    except openai.AuthenticationError:
        raise
    except openai.APIConnectionError:
        raise
    except openai.OpenAIError as err:
        LOGGER.debug("Model listing unavailable on %s: %s", base_url, err)


class QwenConfigFlow(ConfigFlow, domain=DOMAIN):
    """Collect the endpoint, API key and model."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialise the flow."""
        self._data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for the API key, region and model."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._data = dict(user_input)
            if user_input[CONF_REGION] == REGION_CUSTOM:
                return await self.async_step_endpoint()

            errors = await self._async_validate(BASE_URLS[user_input[CONF_REGION]])
            if not errors:
                return await self.async_step_speech()

        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                STEP_USER_SCHEMA, user_input
            ),
            errors=errors,
        )

    async def async_step_endpoint(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for a custom endpoint, e.g. a workspace-scoped host."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._data.update(user_input)
            errors = await self._async_validate(user_input[CONF_BASE_URL])
            if not errors:
                return await self.async_step_speech()

        return self.async_show_form(
            step_id="endpoint",
            data_schema=self.add_suggested_values_to_schema(
                STEP_ENDPOINT_SCHEMA, user_input
            ),
            errors=errors,
            description_placeholders={
                "example": (
                    "https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com"
                    "/compatible-mode/v1"
                )
            },
        )

    async def async_step_speech(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Optionally point the speech platforms at a second region.

        Alibaba does not offer CosyVoice, Qwen-TTS and Qwen-ASR everywhere the
        chat models are available, so an account can need a different endpoint
        — and usually a different key — for speech alone.
        """
        errors: dict[str, str] = {}

        if user_input is not None:
            speech = {key: value for key, value in user_input.items() if value}
            self._data.update(speech)

            if speech.get(CONF_SPEECH_REGION, REGION_SAME) == REGION_SAME:
                return self._create_entry()

            base_url, _, _, api_key = resolve_speech_endpoints(self._data)
            errors = await self._async_validate(base_url, api_key=api_key)
            if not errors:
                return self._create_entry()

        return self.async_show_form(
            step_id="speech",
            data_schema=self.add_suggested_values_to_schema(
                STEP_SPEECH_SCHEMA, user_input
            ),
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Let an existing entry change its endpoints without being deleted."""
        entry = self._get_reconfigure_entry()
        self._data = {
            **dict(entry.data),
            CONF_CHAT_MODEL: entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
        }
        return await self.async_step_speech(user_input)

    async def _async_validate(
        self, base_url: str, api_key: str | None = None
    ) -> dict[str, str]:
        """Return form errors for the collected credentials.

        ``api_key`` defaults to the conversation key; the speech step passes the
        separate key when one was entered.
        """
        try:
            await validate_credentials(
                self.hass, api_key or self._data[CONF_API_KEY], base_url
            )
        except openai.AuthenticationError:
            return {"base": "invalid_auth"}
        except openai.APIConnectionError:
            return {"base": "cannot_connect"}
        except Exception:  # noqa: BLE001
            LOGGER.exception("Unexpected error validating DashScope credentials")
            return {"base": "unknown"}
        return {}

    def _create_entry(self) -> ConfigFlowResult:
        """Store credentials in data and the model in options."""
        data = {
            key: value
            for key, value in self._data.items()
            if key
            in (
                CONF_API_KEY,
                CONF_REGION,
                CONF_BASE_URL,
                CONF_WS_URL,
                CONF_SPEECH_REGION,
                CONF_SPEECH_API_KEY,
                CONF_SPEECH_BASE_URL,
                CONF_SPEECH_WS_URL,
            )
        }
        options = {CONF_CHAT_MODEL: self._data[CONF_CHAT_MODEL]}

        if self.source == "reauth":
            return self.async_update_reload_and_abort(
                self._get_reauth_entry(), data_updates=data
            )

        if self.source == "reconfigure":
            return self.async_update_reload_and_abort(
                self._get_reconfigure_entry(), data=data
            )

        return self.async_create_entry(title=DEFAULT_NAME, data=data, options=options)

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Handle an expired or revoked API key."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for a replacement API key, keeping the existing endpoint."""
        errors: dict[str, str] = {}
        entry = self._get_reauth_entry()

        if user_input is not None:
            self._data = {**dict(entry.data), **user_input}
            self._data[CONF_CHAT_MODEL] = entry.options.get(
                CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL
            )
            base_url, _, _ = resolve_endpoints(self._data)
            errors = await self._async_validate(base_url)
            if not errors:
                return self._create_entry()

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    )
                }
            ),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: QwenConfigEntry) -> OptionsFlow:
        """Return the options flow."""
        return QwenOptionsFlow()


class QwenOptionsFlow(OptionsFlow):
    """Tune the model, the prompt and the voice."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show and store the options."""
        if user_input is not None:
            # An empty selection means "no Home Assistant control".
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            return self.async_create_entry(data=user_input)

        options = self.config_entry.options
        backend = options.get(CONF_TTS_BACKEND, DEFAULT_TTS_BACKEND)
        tts_model = options.get(CONF_TTS_MODEL, DEFAULT_TTS_MODEL)

        if backend == TTS_BACKEND_QWEN:
            tts_models = QWEN_TTS_MODELS
            voices = QWEN_TTS_VOICES
        else:
            tts_models = COSYVOICE_MODELS
            # CosyVoice voices are pinned to a model generation, so only offer
            # the ones matching the model currently selected.
            voices = COSYVOICE_VOICES.get(tts_model, [])

        apis = [
            SelectOptionDict(value=api.id, label=api.name)
            for api in llm.async_get_apis(self.hass)
        ]

        schema = vol.Schema(
            {
                vol.Optional(CONF_PROMPT): TemplateSelector(),
                vol.Optional(CONF_LLM_HASS_API): SelectSelector(
                    SelectSelectorConfig(options=apis, multiple=True)
                ),
                vol.Required(CONF_CHAT_MODEL): _combobox(CHAT_MODELS),
                vol.Required(CONF_MAX_TOKENS): NumberSelector(
                    NumberSelectorConfig(
                        min=1, max=32768, step=1, mode=NumberSelectorMode.BOX
                    )
                ),
                vol.Required(CONF_TEMPERATURE): NumberSelector(
                    NumberSelectorConfig(
                        min=0, max=2, step=0.05, mode=NumberSelectorMode.SLIDER
                    )
                ),
                vol.Required(CONF_TOP_P): NumberSelector(
                    NumberSelectorConfig(
                        min=0, max=1, step=0.05, mode=NumberSelectorMode.SLIDER
                    )
                ),
                vol.Required(CONF_ENABLE_THINKING): bool,
                vol.Required(CONF_FOLLOW_UP): bool,
                vol.Required(CONF_TTS_BACKEND): SelectSelector(
                    SelectSelectorConfig(
                        options=[TTS_BACKEND_COSYVOICE, TTS_BACKEND_QWEN],
                        mode=SelectSelectorMode.DROPDOWN,
                        translation_key="tts_backend",
                    )
                ),
                vol.Required(CONF_TTS_MODEL): _combobox(tts_models),
                vol.Required(CONF_TTS_VOICE): _combobox(voices),
                vol.Required(CONF_STT_BACKEND): SelectSelector(
                    SelectSelectorConfig(
                        options=list(STT_BACKENDS),
                        mode=SelectSelectorMode.DROPDOWN,
                        translation_key="stt_backend",
                    )
                ),
                vol.Required(CONF_STT_MODEL): _combobox(QWEN_ASR_MODELS),
            }
        )

        suggested = {
            CONF_PROMPT: options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT),
            CONF_LLM_HASS_API: options.get(CONF_LLM_HASS_API, []),
            CONF_CHAT_MODEL: options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
            CONF_MAX_TOKENS: options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            CONF_TEMPERATURE: options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            CONF_TOP_P: options.get(CONF_TOP_P, DEFAULT_TOP_P),
            CONF_ENABLE_THINKING: options.get(
                CONF_ENABLE_THINKING, DEFAULT_ENABLE_THINKING
            ),
            CONF_FOLLOW_UP: options.get(CONF_FOLLOW_UP, DEFAULT_FOLLOW_UP),
            CONF_TTS_BACKEND: backend,
            CONF_TTS_MODEL: tts_model,
            CONF_TTS_VOICE: options.get(CONF_TTS_VOICE, DEFAULT_TTS_VOICE),
            CONF_STT_BACKEND: options.get(CONF_STT_BACKEND, DEFAULT_STT_BACKEND),
            CONF_STT_MODEL: options.get(CONF_STT_MODEL, DEFAULT_STT_MODEL),
        }

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(schema, suggested),
        )
