"""Feishu Drive (file upload/download) tools. NOT supported by official Lark MCP."""

from __future__ import annotations

import os

import lark_oapi as lark
from lark_oapi.api.drive.v1 import (
    ListFileRequest,
    UploadAllFileRequest,
    UploadAllFileRequestBody,
    DownloadFileRequest,
)

# Allowed base directories for file operations (security sandbox)
_custom_dir = os.environ.get("FEISHU_FILE_DIR")
_ALLOWED_DIRS = (
    [_custom_dir] if _custom_dir
    else [os.path.expanduser("~/feishu-files"), "/tmp/feishu-files"]
)


def _validate_path(path: str, operation: str) -> str:
    """Validate file path is within allowed directories."""
    abs_path = os.path.realpath(os.path.abspath(path))

    # Allow any path under allowed dirs
    for allowed in _ALLOWED_DIRS:
        allowed = os.path.realpath(os.path.abspath(allowed))
        if abs_path.startswith(allowed + os.sep) or abs_path == allowed:
            return abs_path

    raise PermissionError(
        f"Path '{path}' is outside allowed directories for {operation}. "
        f"Allowed: {_ALLOWED_DIRS}. "
        f"Set FEISHU_FILE_DIR env to customize."
    )

from feishu_mcp.utils import extract_response

TOOLS = [
    {
        "name": "feishu_upload_file",
        "description": "Upload a file to Feishu Drive. NOT supported by official Lark MCP.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Local file path to upload",
                },
                "parent_token": {
                    "type": "string",
                    "description": "Parent folder token in Feishu Drive",
                },
                "file_name": {
                    "type": "string",
                    "description": "Optional custom file name. Defaults to the original file name.",
                },
            },
            "required": ["file_path", "parent_token"],
        },
    },
    {
        "name": "feishu_download_file",
        "description": "Download a file from Feishu Drive to local filesystem. NOT supported by official Lark MCP.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_token": {
                    "type": "string",
                    "description": "File token in Feishu Drive",
                },
                "save_path": {
                    "type": "string",
                    "description": "Local path to save the downloaded file",
                },
            },
            "required": ["file_token", "save_path"],
        },
    },
    {
        "name": "feishu_list_files",
        "description": "List files in a Feishu Drive folder.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "folder_token": {
                    "type": "string",
                    "description": "Folder token. Use empty string for root folder.",
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
]


def handle(client: lark.Client, name: str, args: dict) -> dict | str:
    if name == "feishu_upload_file":
        return _upload_file(client, args)
    elif name == "feishu_download_file":
        return _download_file(client, args)
    elif name == "feishu_list_files":
        return _list_files(client, args)
    raise ValueError(f"Unknown drive tool: {name}")


def _upload_file(client: lark.Client, args: dict) -> dict | str:
    file_path = _validate_path(args["file_path"], "upload")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_name = args.get("file_name") or os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    with open(file_path, "rb") as f:
        req = (
            UploadAllFileRequest.builder()
            .request_body(
                UploadAllFileRequestBody.builder()
                .file_name(file_name)
                .parent_type("explorer")
                .parent_node(args["parent_token"])
                .size(file_size)
                .file(f)
                .build()
            )
            .build()
        )
        resp = client.drive.v1.file.upload_all(req)
    return extract_response(resp)


def _download_file(client: lark.Client, args: dict) -> dict | str:
    req = (
        DownloadFileRequest.builder()
        .file_token(args["file_token"])
        .build()
    )
    resp = client.drive.v1.file.download(req)

    if not resp.success():
        raise RuntimeError(
            f"Feishu API error: code={resp.code}, msg={resp.msg}"
        )

    save_path = _validate_path(args["save_path"], "download")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(resp.file.read())

    return f"File downloaded to {save_path}"


def _list_files(client: lark.Client, args: dict) -> dict | str:
    builder = ListFileRequest.builder().page_size(
        min(args.get("page_size", 20), 200)
    )
    if args.get("folder_token"):
        builder = builder.folder_token(args["folder_token"])
    if args.get("page_token"):
        builder = builder.page_token(args["page_token"])

    resp = client.drive.v1.file.list(builder.build())
    return extract_response(resp)
