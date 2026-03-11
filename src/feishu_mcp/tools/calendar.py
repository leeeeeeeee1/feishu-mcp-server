"""Feishu Calendar tools."""

from __future__ import annotations

import lark_oapi as lark
from lark_oapi.api.calendar.v4 import (
    CreateCalendarEventRequest,
    ListCalendarEventRequest,
    CalendarEvent,
    EventTime,
)

from feishu_mcp.utils import extract_response

TOOLS = [
    {
        "name": "feishu_create_event",
        "description": "Create a calendar event in Feishu.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendar_id": {
                    "type": "string",
                    "description": "Calendar ID. Use 'primary' for the default calendar.",
                    "default": "primary",
                },
                "summary": {
                    "type": "string",
                    "description": "Event title/summary",
                },
                "description": {
                    "type": "string",
                    "description": "Event description",
                },
                "start_time": {
                    "type": "string",
                    "description": "Start time as Unix timestamp (seconds)",
                },
                "end_time": {
                    "type": "string",
                    "description": "End time as Unix timestamp (seconds)",
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone, e.g. Asia/Shanghai",
                    "default": "Asia/Shanghai",
                },
            },
            "required": ["summary", "start_time", "end_time"],
        },
    },
    {
        "name": "feishu_list_events",
        "description": "List calendar events within a time range.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calendar_id": {
                    "type": "string",
                    "default": "primary",
                },
                "start_time": {
                    "type": "string",
                    "description": "Start of range as Unix timestamp (seconds)",
                },
                "end_time": {
                    "type": "string",
                    "description": "End of range as Unix timestamp (seconds)",
                },
                "page_size": {
                    "type": "integer",
                    "default": 20,
                },
                "page_token": {
                    "type": "string",
                },
            },
            "required": ["start_time", "end_time"],
        },
    },
]


def handle(client: lark.Client, name: str, args: dict) -> dict | str:
    if name == "feishu_create_event":
        return _create_event(client, args)
    elif name == "feishu_list_events":
        return _list_events(client, args)
    raise ValueError(f"Unknown calendar tool: {name}")


def _create_event(client: lark.Client, args: dict) -> dict | str:
    calendar_id = args.get("calendar_id", "primary")

    event_builder = CalendarEvent.builder().summary(args["summary"])

    if args.get("description"):
        event_builder = event_builder.description(args["description"])

    start = EventTime.builder().time_stamp(args["start_time"]).build()
    end = EventTime.builder().time_stamp(args["end_time"]).build()
    event_builder = event_builder.start_time(start).end_time(end)

    req = (
        CreateCalendarEventRequest.builder()
        .calendar_id(calendar_id)
        .request_body(event_builder.build())
        .build()
    )
    resp = client.calendar.v4.calendar_event.create(req)
    return extract_response(resp)


def _list_events(client: lark.Client, args: dict) -> dict | str:
    calendar_id = args.get("calendar_id", "primary")
    builder = (
        ListCalendarEventRequest.builder()
        .calendar_id(calendar_id)
        .start_time(args["start_time"])
        .end_time(args["end_time"])
        .page_size(min(args.get("page_size", 20), 100))
    )
    if args.get("page_token"):
        builder = builder.page_token(args["page_token"])

    resp = client.calendar.v4.calendar_event.list(builder.build())
    return extract_response(resp)
