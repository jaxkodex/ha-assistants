"""Conversation agent backed by Qwen on DashScope."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal

import openai
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import QwenConfigEntry
from .const import (
    BACKOFF_MULTIPLIER,
    CONF_CHAT_MODEL,
    CONF_ENABLE_THINKING,
    CONF_FOLLOW_UP,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_CHAT_MODEL,
    DEFAULT_ENABLE_THINKING,
    DEFAULT_FOLLOW_UP,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    INITIAL_BACKOFF,
    LOGGER,
    MAX_RETRIES,
    MAX_TOOL_ITERATIONS,
)
from .entity import QwenBaseEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: QwenConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation entity."""
    async_add_entities([QwenConversationEntity(entry)])


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Render an HA LLM tool as an OpenAI function definition."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
        },
    }


def _convert_content_to_messages(
    content: list[conversation.Content],
) -> list[dict[str, Any]]:
    """Render the chat log as OpenAI chat-completion messages."""
    messages: list[dict[str, Any]] = []

    for item in content:
        if isinstance(item, conversation.SystemContent):
            messages.append({"role": "system", "content": item.content})
        elif isinstance(item, conversation.UserContent):
            messages.append({"role": "user", "content": item.content})
        elif isinstance(item, conversation.AssistantContent):
            message: dict[str, Any] = {
                "role": "assistant",
                "content": item.content or "",
            }
            if item.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(tool_call.tool_args),
                        },
                    }
                    for tool_call in item.tool_calls
                ]
            messages.append(message)
        elif isinstance(item, conversation.ToolResultContent):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.tool_call_id,
                    "content": json.dumps(item.tool_result),
                }
            )

    return messages


async def _transform_stream(
    stream: openai.AsyncStream[Any],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Convert OpenAI chat-completion chunks into chat log deltas.

    The chat log expects dicts: a ``role`` key opens a new assistant message,
    and later ``content`` / ``thinking_content`` values are concatenated onto
    it. Tool-call arguments arrive split across many chunks, so they are
    buffered here and emitted once the model signals it is done.
    """
    started = False
    # tool call index -> partial {id, name, arguments}
    pending: dict[int, dict[str, str]] = {}

    async for chunk in stream:
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = choice.delta

        if not started:
            yield {"role": "assistant"}
            started = True

        if delta.content:
            yield {"content": delta.content}

        # DashScope exposes Qwen's thinking mode as a non-standard field.
        if reasoning := getattr(delta, "reasoning_content", None):
            yield {"thinking_content": reasoning}

        for tool_call in delta.tool_calls or []:
            buffer = pending.setdefault(
                tool_call.index, {"id": "", "name": "", "arguments": ""}
            )
            if tool_call.id:
                buffer["id"] = tool_call.id
            if tool_call.function:
                if tool_call.function.name:
                    buffer["name"] = tool_call.function.name
                if tool_call.function.arguments:
                    buffer["arguments"] += tool_call.function.arguments

        if choice.finish_reason and pending:
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=buffer["id"],
                        tool_name=buffer["name"],
                        tool_args=json.loads(buffer["arguments"] or "{}"),
                    )
                    for buffer in pending.values()
                ]
            }
            pending.clear()


class QwenConversationEntity(conversation.ConversationEntity, QwenBaseEntity):
    """Assist conversation agent powered by Qwen."""

    _attr_supports_streaming = True
    _attr_name = None

    def __init__(self, entry: QwenConfigEntry) -> None:
        """Initialise the agent."""
        QwenBaseEntity.__init__(self, entry)
        self._attr_unique_id = entry.entry_id
        if entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Qwen is multilingual; let Assist route any language here."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """Register as the entry's conversation agent."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister the conversation agent."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Answer a sentence."""
        options = self.entry.options

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(self.entry.domain),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_converse(chat_log)

        result = conversation.async_get_result_from_chat_log(user_input, chat_log)

        # Home Assistant already continues on its own when the reply ends in a
        # question mark; this widens that to every reply so a satellite does
        # not need a second wake word for a follow-up.
        if options.get(CONF_FOLLOW_UP, DEFAULT_FOLLOW_UP):
            result.continue_conversation = True

        return result

    async def _async_converse(self, chat_log: conversation.ChatLog) -> None:
        """Drive the model until it stops asking for tools."""
        options = self.entry.options
        client = self.entry.runtime_data.client
        model = options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)

        tools: list[dict[str, Any]] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        messages = _convert_content_to_messages(chat_log.content)

        for _iteration in range(MAX_TOOL_ITERATIONS):
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                # NumberSelector hands back a float; DashScope rejects 3000.0.
                "max_tokens": int(options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)),
                "temperature": options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                "top_p": options.get(CONF_TOP_P, DEFAULT_TOP_P),
                "stream": True,
                "extra_body": {
                    "enable_thinking": options.get(
                        CONF_ENABLE_THINKING, DEFAULT_ENABLE_THINKING
                    )
                },
            }
            if tools:
                kwargs["tools"] = tools

            stream = await self._async_create_stream(client, kwargs)

            async for _content in chat_log.async_add_delta_content_stream(
                self.entity_id, _transform_stream(stream)
            ):
                pass

            messages = _convert_content_to_messages(chat_log.content)

            # Tools ran while the stream was consumed; if their results are the
            # newest thing in the log, the model still has to turn them into an
            # answer. Otherwise it replied in prose and we are done.
            if not chat_log.unresponded_tool_results:
                return

        raise HomeAssistantError(
            f"Qwen kept requesting tools after {MAX_TOOL_ITERATIONS} rounds"
        )

    async def _async_create_stream(
        self, client: openai.AsyncOpenAI, kwargs: dict[str, Any]
    ) -> openai.AsyncStream[Any]:
        """Open a streaming completion, backing off when throttled."""
        delay = INITIAL_BACKOFF
        last_error: openai.RateLimitError | None = None

        for attempt in range(MAX_RETRIES):
            try:
                return await client.chat.completions.create(**kwargs)
            except openai.AuthenticationError as err:
                raise HomeAssistantError(
                    "DashScope rejected the API key. Re-authenticate the "
                    "Qwen Conversation integration."
                ) from err
            except openai.RateLimitError as err:
                # ``insufficient_quota`` also arrives as a RateLimitError, and
                # retrying an exhausted balance is pointless.
                if getattr(err, "code", None) == "insufficient_quota":
                    raise HomeAssistantError(
                        "The DashScope account has no remaining quota."
                    ) from err
                last_error = err
                if attempt < MAX_RETRIES - 1:
                    LOGGER.debug("DashScope throttled us, retrying in %.1fs", delay)
                    await asyncio.sleep(delay)
                    delay *= BACKOFF_MULTIPLIER
            except openai.OpenAIError as err:
                raise HomeAssistantError(f"Error talking to DashScope: {err}") from err

        raise HomeAssistantError(
            f"DashScope kept throttling after {MAX_RETRIES} attempts: {last_error}"
        )
