# Multi Assitant Conversation for Home Assistant

A custom integration that connects Home Assistant's Assist pipeline to Alibaba
Cloud Model Studio (DashScope). One config entry gives you three entities:

| Platform | Powered by | Notes |
| --- | --- | --- |
| **Conversation** | Qwen via the OpenAI-compatible endpoint | Streaming, with Home Assistant tool calling so it can control your devices |
| **Text-to-speech** | CosyVoice or Qwen-TTS | Selectable provider |
| **Speech-to-text** | Qwen-ASR | ~28 languages, auto-detect |

All inference is remote. Nothing runs locally, and no model is bundled.

---

## Requirements

- Home Assistant **2026.4** or newer
- An Alibaba Cloud Model Studio account and a **DashScope API key**

## Getting a DashScope API key

1. Sign in to [Alibaba Cloud Model Studio](https://bailian.console.aliyun.com/)
   (mainland China) or the
   [international console](https://modelstudio.console.alibabacloud.com/).
2. Activate Model Studio if you have not already.
3. Open **API keys** and create one. Copy it now — it is shown once.

> **API keys are region-specific.** A key created in the Singapore console will
> be rejected by the Beijing endpoint and vice versa. If you get
> `invalid_auth` during setup, this is almost always the cause.

## Choosing a region

| Region | Endpoint | Use when |
| --- | --- | --- |
| Singapore / international | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | You created your key on the international console |
| Beijing / mainland China | `https://dashscope.aliyuncs.com/compatible-mode/v1` | You created your key on the mainland console |
| Custom | whatever you enter | Workspace-scoped or self-hosted deployments |

Alibaba is migrating to per-workspace hostnames of the form
`https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1`.
The two hostnames above are documented as legacy but still functional. If you
have a workspace-scoped URL, choose **Custom** and paste it — the websocket and
plain-HTTP endpoints used by the speech platforms are derived from the same
host automatically, and can be overridden separately if needed.

---

## Installation

### HACS

1. HACS → **⋮** → **Custom repositories**.
2. Add `https://github.com/jvilcayp/ha-alibaba` with category **Integration**.
3. Find **Qwen Conversation** in HACS, install it, and restart Home Assistant.

### Manual install on Home Assistant Green / HAOS

HAOS is an appliance image with no general shell, so copy the files in over the
network:

**Using the Samba add-on (easiest)**

1. Install and start the **Samba share** add-on
   (Settings → Add-ons → Add-on store).
2. From your computer, open `\\homeassistant\config` (Windows) or
   `smb://homeassistant/config` (macOS).
3. Create `custom_components` if it does not exist, then copy the
   `custom_components/qwen_conversation` folder from this repository into it.
4. Restart Home Assistant (Settings → System → **Restart**).

**Using the `ha` CLI**

If you have the SSH add-on or console access:

```sh
cd /config
mkdir -p custom_components
# copy qwen_conversation/ into custom_components/, then:
ha core restart
```

The result should look like:

```
/config/custom_components/qwen_conversation/
├── __init__.py
├── config_flow.py
├── const.py
├── conversation.py
├── entity.py
├── manifest.json
├── stt.py
├── stt_backend.py
├── strings.json
├── translations/en.json
├── tts.py
└── tts_backend.py
```

## Configuration

Everything is configured through the UI; there is no YAML.

1. **Settings → Devices & Services → Add Integration → Qwen Conversation**.
2. Enter your API key, pick the region, and choose a model.
3. Once created, press **Configure** on the entry to set the prompt, enable
   device control, and pick TTS/STT models and voices.

### Options

| Option | Default | Notes |
| --- | --- | --- |
| Instructions | HA default prompt | System prompt, templatable |
| Control Home Assistant | off | Pick an LLM API (e.g. *Assist*) to let the model call devices |
| Conversation model | `qwen3.7-plus` | Any model ID; pinned snapshots like `qwen3.7-plus-2026-04-02` work |
| Temperature / Top P / Max tokens | 1.0 / 1.0 / 3000 | |
| Enable thinking mode | off | More accurate, noticeably slower — usually not worth it for voice |
| Text-to-speech provider | CosyVoice | Or Qwen-TTS |
| Speech-to-text provider | Qwen-ASR | |

### Using it in a voice pipeline

Settings → Voice assistants → create or edit a pipeline, then set
**Conversation agent**, **Speech-to-text** and **Text-to-speech** to the
Qwen Conversation entities.

---

## Notes and gotchas

**CosyVoice voices are tied to a model generation and are not
interchangeable.** `longxiaochun` is the v1-era name; `cosyvoice-v2` needs
`longxiaochun_v2` and `cosyvoice-v3-*` needs `longxiaochun_v3`. Using the wrong
combination produces an error with no audio. The options page narrows the voice
list to the selected model — change the model, save, then reopen the page to
pick a matching voice. The voice field stays free-text so newer models
(`cosyvoice-v3.5-*`), whose voice lists are not published yet, still work.

**Qwen-TTS returns a link, not audio.** The integration downloads it for you;
this costs one extra round trip compared to CosyVoice.

**Rate limits.** DashScope enforces a burst limit on top of the steady-state
rate limit, so bursts can be throttled while you are still under quota. All
three platforms retry with exponential backoff. An exhausted balance
(`insufficient_quota`) fails immediately rather than retrying, since retrying
cannot help.

**Auth failures** raise a re-authentication prompt in Home Assistant rather
than failing silently.

---

## Development

Tests use `pytest-homeassistant-custom-component`, which requires the same
Python version as Home Assistant itself (**3.14.2+** for 2026.7).

```sh
python -m venv .venv
. .venv/bin/activate           # .venv\Scripts\activate on Windows
pip install -r requirements-test.txt
pytest
```

## License

MIT
