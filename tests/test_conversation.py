"""Tests for the Qwen conversation agent."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.qwen_conversation.const import (
    CONF_FOLLOW_UP,
    CONF_FOLLOW_UP_TURNS,
)
from custom_components.qwen_conversation.conversation import (
    _convert_content_to_messages,
    _transform_stream,
)
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent, llm


def _chunk(
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
    reasoning: str | None = None,
):
    """Build a minimal OpenAI streaming chunk."""
    delta = SimpleNamespace(
        content=content, tool_calls=tool_calls, reasoning_content=reasoning
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def _tool_call_delta(index: int, call_id: str = "", name: str = "", args: str = ""):
    """Build a partial tool call as it arrives on the wire."""
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=args),
    )


async def _collect(chunks: list) -> list[dict]:
    """Run the chunks through the transformer."""

    async def stream():
        for chunk in chunks:
            yield chunk

    return [delta async for delta in _transform_stream(stream())]


async def test_transform_stream_text() -> None:
    """Text deltas open one assistant message and stream content."""
    deltas = await _collect(
        [_chunk(content="Hello "), _chunk(content="world"), _chunk(finish_reason="stop")]
    )

    assert deltas == [
        {"role": "assistant"},
        {"content": "Hello "},
        {"content": "world"},
    ]


async def test_transform_stream_thinking() -> None:
    """DashScope's reasoning_content maps onto thinking_content."""
    deltas = await _collect(
        [_chunk(reasoning="hmm"), _chunk(content="hi"), _chunk(finish_reason="stop")]
    )

    assert {"thinking_content": "hmm"} in deltas
    assert deltas[0] == {"role": "assistant"}


async def test_transform_stream_assembles_split_tool_call() -> None:
    """Tool arguments split across chunks are joined into one ToolInput."""
    deltas = await _collect(
        [
            _chunk(tool_calls=[_tool_call_delta(0, "call_1", "turn_on", '{"na')]),
            _chunk(tool_calls=[_tool_call_delta(0, args='me": "kitchen"}')]),
            _chunk(finish_reason="tool_calls"),
        ]
    )

    tool_calls = deltas[-1]["tool_calls"]
    assert len(tool_calls) == 1
    assert isinstance(tool_calls[0], llm.ToolInput)
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].tool_name == "turn_on"
    assert tool_calls[0].tool_args == {"name": "kitchen"}


async def test_transform_stream_parallel_tool_calls() -> None:
    """Two concurrent tool calls are kept apart by their index."""
    deltas = await _collect(
        [
            _chunk(
                tool_calls=[
                    _tool_call_delta(0, "a", "turn_on", "{}"),
                    _tool_call_delta(1, "b", "turn_off", "{}"),
                ]
            ),
            _chunk(finish_reason="tool_calls"),
        ]
    )

    tool_calls = deltas[-1]["tool_calls"]
    assert [call.tool_name for call in tool_calls] == ["turn_on", "turn_off"]


async def test_transform_stream_ignores_empty_choices() -> None:
    """Usage-only chunks carry no choices and must not crash the transform."""
    deltas = await _collect(
        [SimpleNamespace(choices=[]), _chunk(content="hi"), _chunk(finish_reason="stop")]
    )

    assert deltas == [{"role": "assistant"}, {"content": "hi"}]


def test_convert_content_to_messages() -> None:
    """The chat log round-trips into OpenAI message dicts."""
    messages = _convert_content_to_messages(
        [
            conversation.SystemContent(content="be helpful"),
            conversation.UserContent(content="lights on"),
            conversation.AssistantContent(
                agent_id="conversation.qwen",
                content=None,
                tool_calls=[
                    llm.ToolInput(
                        id="call_1", tool_name="turn_on", tool_args={"name": "kitchen"}
                    )
                ],
            ),
            conversation.ToolResultContent(
                agent_id="conversation.qwen",
                tool_call_id="call_1",
                tool_name="turn_on",
                tool_result={"success": True},
            ),
        ]
    )

    assert messages[0] == {"role": "system", "content": "be helpful"}
    assert messages[1] == {"role": "user", "content": "lights on"}

    assistant = messages[2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "turn_on"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "name": "kitchen"
    }

    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": json.dumps({"success": True}),
    }


async def test_entity_is_registered(hass: HomeAssistant, init_integration) -> None:
    """The conversation entity is created and exposed to Assist."""
    state = hass.states.get("conversation.qwen_conversation")
    assert state is not None


async def test_conversation_answers(
    hass: HomeAssistant, init_integration, mock_openai
) -> None:
    """A plain text answer reaches the caller."""

    async def stream():
        yield _chunk(content="The kitchen light is on.")
        yield _chunk(finish_reason="stop")

    mock_openai.chat.completions.create = AsyncMock(return_value=stream())

    result = await conversation.async_converse(
        hass, "is the kitchen light on?", None, None, agent_id="conversation.qwen_conversation"
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert (
        result.response.speech["plain"]["speech"] == "The kitchen light is on."
    )
    # Home Assistant's own rule: a statement does not reopen the microphone.
    assert result.continue_conversation is False


async def test_follow_up_keeps_the_conversation_open(
    hass: HomeAssistant, init_integration, mock_openai
) -> None:
    """With follow-up on, a statement still reopens the microphone."""
    hass.config_entries.async_update_entry(
        init_integration,
        options={**init_integration.options, CONF_FOLLOW_UP: True},
    )
    await hass.async_block_till_done()

    async def stream():
        yield _chunk(content="The kitchen light is on.")
        yield _chunk(finish_reason="stop")

    mock_openai.chat.completions.create = AsyncMock(return_value=stream())

    result = await conversation.async_converse(
        hass,
        "is the kitchen light on?",
        None,
        None,
        agent_id="conversation.qwen_conversation",
    )

    assert result.continue_conversation is True


async def test_follow_up_stops_at_the_turn_limit(
    hass: HomeAssistant, init_integration, mock_openai
) -> None:
    """The wake word is required again once the turn cap is reached.

    Without this the conversation has no exit: every reply reopens the
    microphone and any transcribable noise starts another turn.
    """
    hass.config_entries.async_update_entry(
        init_integration,
        options={
            **init_integration.options,
            CONF_FOLLOW_UP: True,
            CONF_FOLLOW_UP_TURNS: 2,
        },
    )
    await hass.async_block_till_done()

    def reply():
        async def stream():
            yield _chunk(content="Done.")
            yield _chunk(finish_reason="stop")

        return stream()

    conversation_id = None
    seen = []
    for _ in range(3):
        mock_openai.chat.completions.create = AsyncMock(return_value=reply())
        result = await conversation.async_converse(
            hass,
            "turn on the light",
            conversation_id,
            None,
            agent_id="conversation.qwen_conversation",
        )
        conversation_id = result.conversation_id
        seen.append(result.continue_conversation)

    # Two turns carried by one wake word, then it stops asking for more.
    assert seen == [True, False, False]
