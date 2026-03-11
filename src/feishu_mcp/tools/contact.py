"""Feishu Contact (user/department) tools."""

from __future__ import annotations

import lark_oapi as lark
from lark_oapi.api.contact.v3 import (
    GetUserRequest,
    ChildrenDepartmentRequest,
    ListDepartmentRequest,
)

from feishu_mcp.utils import extract_response

TOOLS = [
    {
        "name": "feishu_get_user",
        "description": "Get user information by user ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User ID (open_id, union_id, or user_id)",
                },
                "user_id_type": {
                    "type": "string",
                    "enum": ["open_id", "union_id", "user_id"],
                    "default": "open_id",
                },
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "feishu_list_departments",
        "description": "List top-level departments in the organization.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "parent_department_id": {
                    "type": "string",
                    "description": "Parent department ID. Use '0' for root.",
                    "default": "0",
                },
                "page_size": {
                    "type": "integer",
                    "default": 20,
                },
                "page_token": {
                    "type": "string",
                },
            },
        },
    },
    {
        "name": "feishu_list_department_children",
        "description": "List child departments of a given department.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "department_id": {
                    "type": "string",
                    "description": "Parent department ID",
                },
                "page_size": {
                    "type": "integer",
                    "default": 20,
                },
                "page_token": {
                    "type": "string",
                },
            },
            "required": ["department_id"],
        },
    },
]


def handle(client: lark.Client, name: str, args: dict) -> dict | str:
    if name == "feishu_get_user":
        return _get_user(client, args)
    elif name == "feishu_list_departments":
        return _list_departments(client, args)
    elif name == "feishu_list_department_children":
        return _list_department_children(client, args)
    raise ValueError(f"Unknown contact tool: {name}")


def _get_user(client: lark.Client, args: dict) -> dict | str:
    user_id_type = args.get("user_id_type", "open_id")
    req = (
        GetUserRequest.builder()
        .user_id(args["user_id"])
        .user_id_type(user_id_type)
        .build()
    )
    resp = client.contact.v3.user.get(req)
    return extract_response(resp)


def _list_departments(client: lark.Client, args: dict) -> dict | str:
    parent_id = args.get("parent_department_id", "0")
    builder = (
        ListDepartmentRequest.builder()
        .parent_department_id(parent_id)
        .page_size(min(args.get("page_size", 20), 100))
    )
    if args.get("page_token"):
        builder = builder.page_token(args["page_token"])

    resp = client.contact.v3.department.list(builder.build())
    return extract_response(resp)


def _list_department_children(client: lark.Client, args: dict) -> dict | str:
    builder = (
        ChildrenDepartmentRequest.builder()
        .department_id(args["department_id"])
        .page_size(min(args.get("page_size", 20), 100))
    )
    if args.get("page_token"):
        builder = builder.page_token(args["page_token"])

    resp = client.contact.v3.department.children(builder.build())
    return extract_response(resp)
