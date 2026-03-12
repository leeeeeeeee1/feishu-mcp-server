# CLAUDE.md — feishu-mcp-server

## Project Overview

Feishu MCP Server + Supervisor Hub. Python-based MCP server providing 20 tools across 6 Feishu/Lark modules (IM, Docs, Drive, Bitable, Calendar, Contact), plus a Supervisor Hub with Sonnet-based message routing, task dispatching, and Claude subprocess management.

## Tech Stack

- **Language**: Python 3.9+
- **Framework**: MCP SDK (stdio transport), lark-oapi (Feishu SDK)
- **Bot**: WebSocket long connection via lark-oapi
- **Supervisor**: asyncio + threading (Feishu gateway runs on separate OS thread)
- **Testing**: pytest (no pytest-asyncio — use `asyncio.run()` pattern)
- **Package Manager**: pip / pyproject.toml

## Architecture

```
src/
├── feishu_mcp/        # MCP Server (20 tools)
│   ├── server.py      # Tool registration
│   ├── tools/         # Tool implementations by module
│   ├── auth.py        # Feishu auth
│   └── bot.py         # WebSocket bot
└── supervisor/        # Supervisor Hub
    ├── main.py        # Supervisor class, message routing
    ├── task_dispatcher.py  # Task lifecycle, Claude subprocess mgmt
    ├── router_skill.py     # Sonnet routing prompt
    ├── claude_session.py   # Claude CLI session wrapper
    ├── feishu_gateway.py   # Feishu API gateway
    ├── scheduler.py        # Cron-like task scheduling
    └── session_monitor.py  # Session health monitoring
```

## Key Patterns

- **threading.Lock for _tasks**: Feishu gateway is a separate OS thread, so `threading.Lock` (not `asyncio.Lock`). Never hold lock across `await`.
- **asyncio.wait_for for timeouts**: Streaming workers wrapped with `SUPERVISOR_TASK_TIMEOUT` (default 1800s).
- **Snapshot returns**: All public query functions return copies, not live references.
- **Immutable task updates**: Use `_set_status()` for state transitions.

## Testing

```bash
# Run all tests
pytest tests/ --tb=short

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Individual test
pytest tests/test_task_dispatcher.py -v
```

- Tests use `asyncio.run()` wrapper (NOT `@pytest.mark.asyncio`)
- Mock `asyncio.create_subprocess_exec` for Claude subprocess tests
- Use `_reset()` fixture to clean task state between tests

## Recommended Skills

When working in this project, prefer these skills:

### Mandatory (BLOCKING — must use)
- **product-manager**: Requirements decomposition, user stories, acceptance criteria — BEFORE any code
- **tdd-workflow**: Red-Green-Refactor cycle enforcement — tests BEFORE implementation
- **python-e2e-testing**: E2E tests for complete user flows — BEFORE declaring "done"
- **dev-reflection**: Session-start health check, post-feature retrospective

### Core (use for every change)
- **python-patterns**: Pythonic idioms, type hints, PEP 8
- **python-testing**: pytest patterns, fixtures, mocking
- **security-review**: Input validation, secret management, subprocess security
- **verification-loop**: Post-change verification
- **supervisor-hub**: Supervisor-specific patterns (custom skill)

### Situational
- **api-design**: When adding/changing MCP tools
- **mcp-builder**: MCP server patterns (from example-skills)
- **python-ci-setup**: When setting up CI/CD or pre-commit hooks

## Development Workflow (MANDATORY sequence)

```
1. PM           → product-manager skill: user stories + acceptance criteria
2. Plan         → planner agent: implementation plan (if 3+ files)
3. E2E Tests    → python-e2e-testing: write failing E2E test for the user flow
4. Unit Tests   → tdd-workflow: write failing unit tests for functions
5. Implement    → minimal code to pass tests (GREEN)
6. Verify       → pytest tests/ --tb=short (0 failures, 0 new warnings)
7. Review       → python-reviewer agent
8. Fix          → address review findings
9. Re-verify    → run tests again
10. Reflect     → dev-reflection: retrospective
```

Skipping steps 1, 3, 4, 6, 7, or 10 is a workflow violation.

## Known Tech Debt (from reflection 2026-03-12)

| Issue | Severity | File | Status |
|-------|----------|------|--------|
| task_dispatcher.py 1309 lines | CRITICAL | task_dispatcher.py | Needs split |
| main.py 1093 lines | CRITICAL | main.py | Needs split |
| 33 pytest ResourceWarnings | HIGH | tests/*.py | Unfixed |
| 12 silent `pass` in except blocks | HIGH | task_dispatcher.py | Unfixed |
| No CI/CD pipeline | HIGH | .github/ missing | Not started |
| No pre-commit hooks | HIGH | .pre-commit-config.yaml | Not started |
| Dependencies unpinned (bare >=) | HIGH | pyproject.toml | Unfixed |
| Coverage 71% (target 80%) | MEDIUM | — | Below target |
| fix/p0 branch not merged to master | MEDIUM | git | Pending PR |
| Hardcoded pattern matching | MEDIUM | task_dispatcher.py:33-113 | By design (P1) |

## Quality Metrics

```
Tests: 602 passing, 0 failures, 33 warnings
Coverage: 71% (target: 80%)
Max file: 1309 lines (limit: 800)
Silent errors: 12 instances
Unmerged branches: 1
```

## Development Notes

- lark-oapi SDK uses snake_case with underscores (e.g. `time_stamp` not `timestamp`)
- WebSocket long connection required for Feishu bot input box to appear
- `claude -p` subprocess needs `CLAUDECODE` env var unset
- GitHub remote: `git@github.com:leeeeeeeee1/feishu-mcp-server.git`

## Mandatory Pre-Work Checklist

Before ANY code change, verify:
1. `git status` — clean working tree, correct branch
2. `wc -l` on target file — if > 600, plan split first
3. After change: `pytest tests/ --tb=short` — 0 failures, no new warnings
4. After change: run python-reviewer agent
5. Commits: one logical change per commit, conventional format
