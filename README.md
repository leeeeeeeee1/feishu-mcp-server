# Feishu MCP Server

A Python MCP (Model Context Protocol) server for Feishu/Lark, enabling Claude Code to interact with Feishu workspace.

## Features

**18 tools across 6 modules:**

| Module | Tools | Description |
|--------|-------|-------------|
| **IM** | send_message, reply_message, list_messages, list_chats | Messaging |
| **Docs** | create_document, get_document, get_document_content, list_document_blocks, edit_document | Cloud documents |
| **Drive** | upload_file, download_file, list_files | File operations |
| **Calendar** | create_event, list_events | Calendar |
| **Contact** | get_user, list_departments, list_department_children | Organization |
| **Bitable** | bitable_create_record, bitable_search_records, bitable_update_record | Multi-dimensional tables |

**Advantages over official `@larksuiteoapi/lark-mcp`:**
- File upload/download support
- Cloud document block editing
- Pure Python (no Node.js dependency)

## Setup

### 1. Create Feishu App

1. Go to [Feishu Open Platform](https://open.feishu.cn/app)
2. Create an app, get `App ID` and `App Secret`
3. Enable required API permissions:
   - `im:message` / `im:chat` (messaging)
   - `docx:document` (documents)
   - `drive:drive` (files)
   - `calendar:calendar` (calendar)
   - `contact:user.base:readonly` / `contact:department.base:readonly` (contacts)
   - `bitable:app` (bitable)

### 2. Install

```bash
cd feishu-mcp-server
pip install -e .
```

### 3. Register with Claude Code

```bash
claude mcp add feishu -e FEISHU_APP_ID=your_app_id -e FEISHU_APP_SECRET=your_app_secret -- python -m feishu_mcp.server
```

For Lark international:
```bash
claude mcp add feishu \
  -e FEISHU_APP_ID=your_app_id \
  -e FEISHU_APP_SECRET=your_app_secret \
  -e FEISHU_DOMAIN=https://open.larksuite.com \
  -- python -m feishu_mcp.server
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FEISHU_APP_ID` | Yes | App ID from Feishu Open Platform |
| `FEISHU_APP_SECRET` | Yes | App Secret |
| `FEISHU_DOMAIN` | No | Set to `https://open.larksuite.com` for Lark international |

## License

MIT
