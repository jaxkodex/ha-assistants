"""Constants for the Qwen Conversation integration."""

from __future__ import annotations

import logging
from typing import Final

LOGGER: Final = logging.getLogger(__package__)

DOMAIN: Final = "qwen_conversation"

DEFAULT_NAME: Final = "Qwen Conversation"

# --- Region / endpoint -------------------------------------------------------
# The bare ``dashscope*.aliyuncs.com`` hosts are documented as legacy but are
# still fully functional. Newer deployments use a workspace-scoped host of the
# form ``https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/...`` which the
# user can enter through the "custom" region option.

CONF_REGION: Final = "region"
CONF_BASE_URL: Final = "base_url"
CONF_WS_URL: Final = "ws_url"

REGION_INTL: Final = "intl"
REGION_CN: Final = "cn"
REGION_CUSTOM: Final = "custom"

REGIONS: Final = [REGION_INTL, REGION_CN, REGION_CUSTOM]

# OpenAI-compatible ("compatible-mode") endpoints, per region.
BASE_URLS: Final[dict[str, str]] = {
    REGION_INTL: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    REGION_CN: "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

# DashScope websocket endpoints used by the CosyVoice TTS backend.
WS_URLS: Final[dict[str, str]] = {
    REGION_INTL: "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference",
    REGION_CN: "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
}

# Plain HTTP endpoints, used by the Qwen-TTS backend.
HTTP_URLS: Final[dict[str, str]] = {
    REGION_INTL: "https://dashscope-intl.aliyuncs.com/api/v1",
    REGION_CN: "https://dashscope.aliyuncs.com/api/v1",
}

DEFAULT_REGION: Final = REGION_INTL

# --- Speech endpoint override ------------------------------------------------
# Alibaba does not serve the same catalogue in every region. An account whose
# chat models live in one region can find that CosyVoice, Qwen-TTS and Qwen-ASR
# are only offered in another, where they need a key issued for that region —
# the symptom is ``Model not exist`` from every speech model while the
# conversation agent works. These settings point the two speech platforms at a
# second endpoint. When they are unset the speech platforms reuse the
# conversation endpoint and key, which is the previous behaviour.

CONF_SPEECH_REGION: Final = "speech_region"
CONF_SPEECH_API_KEY: Final = "speech_api_key"
CONF_SPEECH_BASE_URL: Final = "speech_base_url"
CONF_SPEECH_WS_URL: Final = "speech_ws_url"

# Sentinel meaning "use whatever the conversation agent uses".
REGION_SAME: Final = "same"

SPEECH_REGIONS: Final = [REGION_SAME, REGION_INTL, REGION_CN, REGION_CUSTOM]

DEFAULT_SPEECH_REGION: Final = REGION_SAME

# --- Conversation options ----------------------------------------------------

CONF_CHAT_MODEL: Final = "chat_model"
CONF_PROMPT: Final = "prompt"
CONF_MAX_TOKENS: Final = "max_tokens"
CONF_TEMPERATURE: Final = "temperature"
CONF_TOP_P: Final = "top_p"
CONF_ENABLE_THINKING: Final = "enable_thinking"
CONF_RECOMMENDED: Final = "recommended"

# Verified as available through DashScope compatible-mode as of 2026-07.
# The selector is a combobox, so unlisted or pinned dated snapshots such as
# ``qwen3.6-flash-2026-04-16`` can also be entered by hand.
CHAT_MODELS: Final[list[str]] = [
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen3.6-flash",
]

DEFAULT_CHAT_MODEL: Final = "qwen3.7-plus"
DEFAULT_MAX_TOKENS: Final = 3000
DEFAULT_TEMPERATURE: Final = 1.0
DEFAULT_TOP_P: Final = 1.0
DEFAULT_ENABLE_THINKING: Final = False

# Number of assistant turns we allow while the model is still calling tools,
# before giving up. Guards against a model that loops on tool calls forever.
MAX_TOOL_ITERATIONS: Final = 10

# --- TTS options -------------------------------------------------------------

CONF_TTS_BACKEND: Final = "tts_backend"
CONF_TTS_MODEL: Final = "tts_model"
CONF_TTS_VOICE: Final = "tts_voice"

TTS_BACKEND_COSYVOICE: Final = "cosyvoice"
TTS_BACKEND_QWEN: Final = "qwen_tts"

DEFAULT_TTS_BACKEND: Final = TTS_BACKEND_COSYVOICE
DEFAULT_TTS_MODEL: Final = "cosyvoice-v3-flash"
DEFAULT_TTS_VOICE: Final = "longxiaochun_v3"

# CosyVoice voices are version-suffixed and are NOT interchangeable between
# model versions: ``longxiaochun`` is the v1-era name, ``longxiaochun_v2``
# belongs to cosyvoice-v2 and ``longxiaochun_v3`` to cosyvoice-v3-*. Picking a
# voice from the wrong generation is a server-side error, so the options flow
# narrows the voice list to the selected model.
COSYVOICE_MODELS: Final[list[str]] = [
    "cosyvoice-v3.5-plus",
    "cosyvoice-v3.5-flash",
    "cosyvoice-v3-plus",
    "cosyvoice-v3-flash",
    "cosyvoice-v2",
    "cosyvoice-v1",
]

COSYVOICE_VOICES: Final[dict[str, list[str]]] = {
    "cosyvoice-v3-flash": [
        "longxiaochun_v3",
        "longanhuan_v3",
        "longhuhu_v3",
        "longyumi_v3",
        "longanyang",
    ],
    "cosyvoice-v3-plus": [
        "longanhuan",
        "longanyang",
    ],
    "cosyvoice-v2": [
        "longxiaochun_v2",
        "longyumi_v2",
    ],
    "cosyvoice-v1": [
        "longxiaochun",
    ],
}
"""Known-good voices per CosyVoice model.

The v3.5 voice lists are not published in the documentation we could verify, so
they are absent here on purpose; the voice selector stays a free-text combobox
so those models remain usable.
"""

QWEN_TTS_MODELS: Final[list[str]] = [
    "qwen3-tts-flash",
    "qwen3-tts-instruct-flash",
    "qwen-tts-latest",
    "qwen-tts",
]

# Qwen-TTS uses English voice names shared across the qwen*-tts family.
QWEN_TTS_VOICES: Final[list[str]] = [
    "Cherry",
    "Serena",
    "Ethan",
    "Chelsie",
    "Momo",
    "Vivian",
    "Moon",
    "Maia",
    "Kai",
    "Nofish",
    "Jada",
    "Dylan",
    "Sunny",
]

# --- STT options -------------------------------------------------------------

CONF_STT_BACKEND: Final = "stt_backend"
CONF_STT_MODEL: Final = "stt_model"

STT_BACKEND_QWEN: Final = "qwen_asr"

DEFAULT_STT_BACKEND: Final = STT_BACKEND_QWEN
DEFAULT_STT_MODEL: Final = "qwen3-asr-flash"

# Qwen-ASR is reachable through the same OpenAI-compatible chat endpoint as the
# conversation agent, taking the audio as an inline base64 data URI. There is no
# ``/audio/transcriptions`` route on DashScope.
QWEN_ASR_MODELS: Final[list[str]] = [
    "qwen3-asr-flash",
    "qwen3-asr-flash-2026-02-10",
    "qwen3-asr-flash-2025-09-08",
]

# Languages Qwen-ASR can transcribe. Assist passes a full locale such as
# ``en-US``; only the leading subtag is sent to DashScope.
STT_LANGUAGES: Final[list[str]] = [
    "zh-CN",
    "zh-HK",
    "zh-TW",
    "en-US",
    "en-GB",
    "ja-JP",
    "ko-KR",
    "de-DE",
    "ru-RU",
    "fr-FR",
    "pt-PT",
    "ar-SA",
    "it-IT",
    "es-ES",
    "hi-IN",
    "id-ID",
    "th-TH",
    "tr-TR",
    "uk-UA",
    "vi-VN",
    "cs-CZ",
    "da-DK",
    "fil-PH",
    "fi-FI",
    "is-IS",
    "ms-MY",
    "nb-NO",
    "pl-PL",
    "sv-SE",
]

# --- Retry behaviour ---------------------------------------------------------

# DashScope enforces both a steady-state rate limit and a separate burst limit
# (``Throttling.BurstRate``), so a spiky client can be throttled while still
# under quota. Back off rather than retrying tightly.
MAX_RETRIES: Final = 3
INITIAL_BACKOFF: Final = 1.0
BACKOFF_MULTIPLIER: Final = 2.0
