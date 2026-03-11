"""Feishu Cloud Document tools."""

from __future__ import annotations

import json

import lark_oapi as lark
from lark_oapi.api.docx.v1 import (
    CreateDocumentBlockChildrenRequest,
    CreateDocumentBlockChildrenRequestBody,
    CreateDocumentRequest,
    CreateDocumentRequestBody,
    GetDocumentRequest,
    ListDocumentBlockRequest,
    RawContentDocumentRequest,
)

from feishu_mcp.utils import extract_response

TOOLS = [
    {
        "name": "feishu_create_document",
        "description": "Create a new Feishu cloud document.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Document title",
                },
                "folder_token": {
                    "type": "string",
                    "description": "Optional folder token to create the document in",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "feishu_get_document",
        "description": "Get metadata of a Feishu document.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID",
                },
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "feishu_get_document_content",
        "description": "Get the plain text content of a Feishu document.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID",
                },
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "feishu_list_document_blocks",
        "description": "List all blocks in a Feishu document. Returns block structure for editing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID",
                },
                "page_size": {
                    "type": "integer",
                    "default": 50,
                },
                "page_token": {
                    "type": "string",
                },
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "feishu_edit_document",
        "description": "Add content blocks to a Feishu document. Supports text, heading, code, list, and other block types. This is a capability the official Lark MCP does NOT support.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID",
                },
                "block_id": {
                    "type": "string",
                    "description": "Parent block ID to insert children into. Use document_id for root.",
                },
                "children": {
                    "type": "string",
                    "description": 'JSON array of block objects. Example: [{"block_type": 2, "text": {"elements": [{"text_run": {"content": "Hello"}}]}}]',
                },
                "index": {
                    "type": "integer",
                    "description": "Position to insert at (-1 for end)",
                    "default": -1,
                },
            },
            "required": ["document_id", "block_id", "children"],
        },
    },
]


def handle(client: lark.Client, name: str, args: dict) -> dict | str:
    if name == "feishu_create_document":
        return _create_document(client, args)
    elif name == "feishu_get_document":
        return _get_document(client, args)
    elif name == "feishu_get_document_content":
        return _get_document_content(client, args)
    elif name == "feishu_list_document_blocks":
        return _list_document_blocks(client, args)
    elif name == "feishu_edit_document":
        return _edit_document(client, args)
    raise ValueError(f"Unknown docs tool: {name}")


def _create_document(client: lark.Client, args: dict) -> dict | str:
    body_builder = CreateDocumentRequestBody.builder().title(args["title"])
    if args.get("folder_token"):
        body_builder = body_builder.folder_token(args["folder_token"])

    req = (
        CreateDocumentRequest.builder()
        .request_body(body_builder.build())
        .build()
    )
    resp = client.docx.v1.document.create(req)
    return extract_response(resp)


def _get_document(client: lark.Client, args: dict) -> dict | str:
    req = GetDocumentRequest.builder().document_id(args["document_id"]).build()
    resp = client.docx.v1.document.get(req)
    return extract_response(resp)


def _get_document_content(client: lark.Client, args: dict) -> dict | str:
    req = (
        RawContentDocumentRequest.builder()
        .document_id(args["document_id"])
        .build()
    )
    resp = client.docx.v1.document.raw_content(req)
    return extract_response(resp)


def _list_document_blocks(client: lark.Client, args: dict) -> dict | str:
    builder = (
        ListDocumentBlockRequest.builder()
        .document_id(args["document_id"])
        .page_size(min(args.get("page_size", 50), 500))
    )
    if args.get("page_token"):
        builder = builder.page_token(args["page_token"])

    resp = client.docx.v1.document_block.list(builder.build())
    return extract_response(resp)


def _edit_document(client: lark.Client, args: dict) -> dict | str:
    children_data = args["children"]
    if isinstance(children_data, str):
        try:
            children_data = json.loads(children_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in 'children': {e}")

    from lark_oapi.api.docx.v1 import Block

    blocks = []
    for child in children_data:
        if not isinstance(child, dict):
            raise ValueError(f"Each child must be a dict, got {type(child)}")
        # Use lark SDK's JSON deserialization
        block_json = json.dumps(child)
        block = Block()
        for k, v in child.items():
            if hasattr(block, k) and not k.startswith("_"):
                setattr(block, k, v)
        blocks.append(block)

    body = (
        CreateDocumentBlockChildrenRequestBody.builder()
        .children(blocks)
        .index(args.get("index", -1))
        .build()
    )

    req = (
        CreateDocumentBlockChildrenRequest.builder()
        .document_id(args["document_id"])
        .block_id(args["block_id"])
        .request_body(body)
        .build()
    )
    resp = client.docx.v1.document_block_children.create(req)
    return extract_response(resp)
