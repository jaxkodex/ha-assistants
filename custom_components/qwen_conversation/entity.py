"""Shared base entity for the Qwen Conversation platforms."""

from __future__ import annotations

from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import Entity

from . import QwenConfigEntry
from .const import DEFAULT_NAME, DOMAIN


class QwenBaseEntity(Entity):
    """Common device info for every entity created from a config entry."""

    _attr_has_entity_name = True

    def __init__(self, entry: QwenConfigEntry) -> None:
        """Attach the entity to the entry's service device."""
        self.entry = entry
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title or DEFAULT_NAME,
            manufacturer="Alibaba Cloud",
            model="DashScope",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
