# Feishu MCP Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-1.0-purple.svg)](https://modelcontextprotocol.io)

A Python MCP (Model Context Protocol) server for Feishu/Lark, enabling Claude Code to interact with Feishu workspace. Includes a **Supervisor Hub** — an intelligent bot that routes messages, manages tasks, and monitors system resources.

## Architecture

```
                         +-----------------------+
                         |     Feishu/Lark       |
                         +----------+------------+
                                    |
                    +---------+-----+---------+
                    |                         |
          +---------v--------+     +----------v---------+
          |   MCP Server     |     |   Supervisor Hub   |
          |  (Claude Code    |     |  (Feishu Bot +     |
          |   integration)   |     |   Task Routing)    |
          +------------------+     +--------------------+
          | 20 tools across  |     | Sonnet Router      |
          | 6 modules        |     | Task Dispatcher    |
          | - IM             |     | Session Manager    |
          | - Docs           |     | Health Monitor     |
          | - Drive          |     | Scheduler          |
          | - Bitable        |     +--------------------+
          | - Calendar       |              |
          | - Contact        |     +--------v-----------+
          +------------------+     |  Claude CLI Workers |
                                   |  (claude -p)        |
                                   +--------------------+
```

## Features

### MCP Server — 20 Tools across 6 Modules

| Module | Tools | Description |
|--------|-------|-------------|
| **IM** | `send_message`, `reply_message`, `list_messages`, `list_chats` | Send text/rich text/cards, list conversations |
| **Docs** | `create_document`, `get_document`, `get_document_content`, `list_document_blocks`, `edit_document` | Cloud document CRUD with block-level editing |
| **Drive** | `upload_file`, `download_file`, `list_files` | File upload/download with path sandboxing |
| **Bitable** | `create_record`, `search_records`, `update_record` | Multi-dimensional table operations with filtering/sorting |
| **Calendar** | `create_event`, `list_events` | Calendar events with timezone support |
| **Contact** | `get_user`, `list_departments`, `list_department_children` | User lookup and org hierarchy navigation |

**Advantages over official `@larksuiteoapi/lark-mcp`:**
- File upload/download support
- Cloud document block-level editing
- Pure Python (no Node.js dependency)
- Supervisor Hub for intelligent task routing

### Supervisor Hub — Intelligent Bot System

A multi-component system that turns Feishu into an AI-powered development interface:

- **Smart Routing**: Sonnet classifies messages and decides the optimal action
- **Task Dispatch**: Manages concurrent Claude CLI workers with session persistence
- **Health Monitoring**: CPU, memory, disk, GPU tracking with Feishu alerts
- **Auto-Reload**: File watcher restarts on code changes
- **Session Continuity**: Tasks survive restarts via session resumption

**Routing Actions:**

| Action | When | Example |
|--------|------|---------|
| `reply` | Knowledge, greetings, planning | "MCP是什么?" |
| `dispatch` | Needs execution/file access | "帮我写个脚本" |
| `dispatch_multi` | Complex multi-part tasks | "重构这三个模块" |
| `follow_up` | Continue existing task | "上个任务再加个功能" |
| `close` | Task completion | "关闭任务" |

**Local Commands:**

| Command | Description |
|---------|-------------|
| `/status` | System status (CPU, memory, GPU, tasks) |
| `/tasks` | List all tasks with status |
| `/gpu` | GPU memory and utilization |
| `/daemons` | Background daemon status |
| `/stop <id>` | Stop a running task |
| `/close <id>` | Close a completed task |
| `/help` | Show all commands |

## Quick Start

### 1. Create Feishu App

1. Go to [Feishu Open Platform](https://open.feishu.cn/app)
2. Create an app, get **App ID** and **App Secret**
3. Enable required permissions:

| Scope | For |
|-------|-----|
| `im:message`, `im:chat` | Messaging |
| `docx:document` | Documents |
| `drive:drive` | Files |
| `calendar:calendar` | Calendar |
| `contact:user.base:readonly`, `contact:department.base:readonly` | Contacts |
| `bitable:app` | Bitable |

4. For Supervisor: Enable **Bot** capability and subscribe to `im.message.receive_v1` event

### 2. Install

```bash
cd feishu-mcp-server
pip install -e .
```

### 3. Use as MCP Server (Claude Code integration)

```bash
claude mcp add feishu \
  -e FEISHU_APP_ID=your_app_id \
  -e FEISHU_APP_SECRET=your_app_secret \
  -- python -m feishu_mcp.server
```

For Lark (international):
```bash
claude mcp add feishu \
  -e FEISHU_APP_ID=your_app_id \
  -e FEISHU_APP_SECRET=your_app_secret \
  -e FEISHU_DOMAIN=https://open.larksuite.com \
  -- python -m feishu_mcp.server
```

### 4. Run Supervisor Hub (Feishu Bot)

```bash
export FEISHU_APP_ID=your_app_id
export FEISHU_APP_SECRET=your_app_secret

# Direct run
python -m supervisor

# With auto-reload daemon (recommended)
bash run-supervisor.sh
```

## Configuration

### MCP Server

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FEISHU_APP_ID` | Yes | — | Feishu app ID |
| `FEISHU_APP_SECRET` | Yes | — | Feishu app secret |
| `FEISHU_DOMAIN` | No | `https://open.feishu.cn` | Use `https://open.larksuite.com` for Lark |
| `FEISHU_FILE_DIR` | No | — | Restrict file operations to this directory |

### Supervisor Hub

| Variable | Default | Description |
|----------|---------|-------------|
| `SUPERVISOR_MAX_WORKERS` | `3` | Max concurrent task workers |
| `SUPERVISOR_MAX_DAEMONS` | `2` | Max persistent background daemons |
| `SUPERVISOR_CLAUDE_MODEL` | `opus` | Claude CLI model for tasks |
| `SUPERVISOR_HEALTH_INTERVAL` | `300` | Health check interval (seconds) |
| `SUPERVISOR_SESSION_DIGEST_INTERVAL` | `900` | Session digest interval (seconds) |
| `SUPERVISOR_DAILY_REPORT_INTERVAL` | `86400` | Daily report interval (seconds) |
| `SUPERVISOR_CPU_THRESHOLD` | `90` | CPU alert threshold (%) |
| `SUPERVISOR_MEMORY_THRESHOLD` | `90` | Memory alert threshold (%) |
| `SUPERVISOR_DISK_THRESHOLD` | `90` | Disk alert threshold (%) |
| `SUPERVISOR_GPU_MEMORY_THRESHOLD` | `95` | GPU memory alert threshold (%) |
| `SUPERVISOR_TASKS_FILE` | `/tmp/supervisor-tasks.json` | Task persistence file |

## Project Structure

```
src/
├── feishu_mcp/              # MCP Server
│   ├── server.py            # MCP entry point
│   ├── auth.py              # Token management
│   ├── bot.py               # Simple forwarding bot
│   ├── utils.py             # Response helpers
│   └── tools/               # 6 tool modules
│       ├── im.py            # Messaging (4 tools)
│       ├── docs.py          # Documents (5 tools)
│       ├── drive.py         # Files (3 tools)
│       ├── bitable.py       # Tables (3 tools)
│       ├── calendar.py      # Events (2 tools)
│       └── contact.py       # Contacts (3 tools)
│
└── supervisor/              # Supervisor Hub
    ├── main.py              # Orchestrator & command handler
    ├── router_skill.py      # Sonnet message classifier
    ├── task_dispatcher.py   # Worker & daemon management
    ├── claude_session.py    # Claude CLI wrapper
    ├── feishu_gateway.py    # WebSocket + push messaging
    ├── scheduler.py         # Periodic health/digest/reports
    ├── session_monitor.py   # Claude session tracking
    └── container_monitor.py # System resource monitoring

tests/                       # 10 test modules
run-supervisor.sh            # Auto-reload daemon script
run-bot.sh                   # Bot daemon script
feishu-bot.service           # systemd unit file
```

## Development

### Run Tests

```bash
pytest tests/ --tb=short
```

### Auto-Reload

Both `run-bot.sh` and `run-supervisor.sh` use [watchdog](https://github.com/gorakhargosh/watchdog) to monitor `src/` for Python file changes and automatically restart the process.

### systemd Service

```bash
sudo cp feishu-bot.service /etc/systemd/system/
sudo systemctl enable feishu-bot
sudo systemctl start feishu-bot
```

## Dependencies

- [mcp](https://pypi.org/project/mcp/) >= 1.0.0 — Model Context Protocol SDK
- [lark-oapi](https://pypi.org/project/lark-oapi/) >= 1.4.0 — Official Feishu/Lark SDK
- [psutil](https://pypi.org/project/psutil/) >= 5.9.0 — System monitoring
- [watchdog](https://pypi.org/project/watchdog/) >= 3.0.0 — File change detection

Python >= 3.10 required.

## License

MIT
