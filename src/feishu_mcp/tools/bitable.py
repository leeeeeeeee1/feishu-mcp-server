"""Feishu Bitable (multi-dimensional table) tools."""

from __future__ import annotations

import json

import lark_oapi as lark
from lark_oapi.api.bitable.v1 import (
    CreateAppTableRecordRequest,
    AppTableRecord,
    SearchAppTableRecordRequest,
    SearchAppTableRecordRequestBody,
    UpdateAppTableRecordRequest,
)

from feishu_mcp.utils import extract_response

TOOLS = [
    {
        "name": "feishu_bitable_create_record",
        "description": "Create a new record in a Feishu Bitable (multi-dimensional table).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "app_token": {
                    "type": "string",
                    "description": "Bitable app token",
                },
                "table_id": {
                    "type": "string",
                    "description": "Table ID within the Bitable",
                },
                "fields": {
                    "type": "string",
                    "description": 'JSON object of field name to value mappings. Example: {"Name": "John", "Age": 30}',
                },
            },
            "required": ["app_token", "table_id", "fields"],
        },
    },
    {
        "name": "feishu_bitable_search_records",
        "description": "Search records in a Feishu Bitable with optional filter and sort.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "app_token": {
                    "type": "string",
                    "description": "Bitable app token",
                },
                "table_id": {
                    "type": "string",
                    "description": "Table ID",
                },
                "filter": {
                    "type": "string",
                    "description": 'Optional filter JSON. Example: {"conjunction":"and","conditions":[{"field_name":"Status","operator":"is","value":["Done"]}]}',
                },
                "sort": {
                    "type": "string",
                    "description": 'Optional sort JSON array. Example: [{"field_name":"Created","desc":true}]',
                },
                "page_size": {
                    "type": "integer",
                    "default": 20,
                },
                "page_token": {
                    "type": "string",
                },
            },
            "required": ["app_token", "table_id"],
        },
    },
    {
        "name": "feishu_bitable_update_record",
        "description": "Update an existing record in a Feishu Bitable.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "app_token": {
                    "type": "string",
                    "description": "Bitable app token",
                },
                "table_id": {
                    "type": "string",
                    "description": "Table ID",
                },
                "record_id": {
                    "type": "string",
                    "description": "Record ID to update",
                },
                "fields": {
                    "type": "string",
                    "description": 'JSON object of field name to new value mappings',
                },
            },
            "required": ["app_token", "table_id", "record_id", "fields"],
        },
    },
]


def handle(client: lark.Client, name: str, args: dict) -> dict | str:
    if name == "feishu_bitable_create_record":
        return _create_record(client, args)
    elif name == "feishu_bitable_search_records":
        return _search_records(client, args)
    elif name == "feishu_bitable_update_record":
        return _update_record(client, args)
    raise ValueError(f"Unknown bitable tool: {name}")


def _create_record(client: lark.Client, args: dict) -> dict | str:
    fields = args["fields"]
    if isinstance(fields, str):
        try:
            fields = json.loads(fields)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in 'fields': {e}")

    record = AppTableRecord.builder().fields(fields).build()
    req = (
        CreateAppTableRecordRequest.builder()
        .app_token(args["app_token"])
        .table_id(args["table_id"])
        .request_body(record)
        .build()
    )
    resp = client.bitable.v1.app_table_record.create(req)
    return extract_response(resp)


def _search_records(client: lark.Client, args: dict) -> dict | str:
    body_builder = SearchAppTableRecordRequestBody.builder().page_size(
        min(args.get("page_size", 20), 100)
    )

    if args.get("filter"):
        filter_data = args["filter"]
        if isinstance(filter_data, str):
            filter_data = json.loads(filter_data)
        body_builder = body_builder.filter(filter_data)

    if args.get("sort"):
        sort_data = args["sort"]
        if isinstance(sort_data, str):
            sort_data = json.loads(sort_data)
        body_builder = body_builder.sort(sort_data)

    builder = (
        SearchAppTableRecordRequest.builder()
        .app_token(args["app_token"])
        .table_id(args["table_id"])
        .request_body(body_builder.build())
    )
    if args.get("page_token"):
        builder = builder.page_token(args["page_token"])

    resp = client.bitable.v1.app_table_record.search(builder.build())
    return extract_response(resp)


def _update_record(client: lark.Client, args: dict) -> dict | str:
    fields = args["fields"]
    if isinstance(fields, str):
        try:
            fields = json.loads(fields)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in 'fields': {e}")

    record = AppTableRecord.builder().fields(fields).build()
    req = (
        UpdateAppTableRecordRequest.builder()
        .app_token(args["app_token"])
        .table_id(args["table_id"])
        .record_id(args["record_id"])
        .request_body(record)
        .build()
    )
    resp = client.bitable.v1.app_table_record.update(req)
    return extract_response(resp)
