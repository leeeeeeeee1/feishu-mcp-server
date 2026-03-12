# CLAUDE.md — feishu-mcp-server

## Project Vision

**AI Butler (大管家)** — not just an MCP server. This is a conversational AI assistant that:
- Understands user intent through natural dialogue
- Intelligently dispatches tasks to the best model + optimal skill
- Manages task lifecycle: create → track → report → close
- Serves a single user (the owner) with full autonomy

**Hierarchy:** Butler (core concept) > Supervisor Hub (implementation) > MCP Server (one feature)

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
    ├── main.py             # Supervisor class, message routing (445 lines)
    ├── task_dispatcher.py  # Task dispatch + close + recover (579 lines, re-exports)
    ├── task_state.py       # Task dataclass, singleton state, semaphores (122 lines)
    ├── task_queries.py     # Read-only task queries (88 lines)
    ├── action_handlers.py  # Sonnet action handlers (213 lines)
    ├── notification.py     # Task result notifications (109 lines)
    ├── route_parser.py     # Route response parsing (210 lines)
    ├── claude_session.py   # Claude CLI session wrapper (458 lines)
    ├── router_skill.py     # Sonnet routing prompt
    ├── command_handlers.py # /command handlers
    ├── subprocess_runner.py # Claude subprocess execution
    ├── feishu_gateway.py   # Feishu API gateway
    ├── task_persistence.py # Task save/load to disk
    ├── task_formatting.py  # Task display formatting
    ├── patterns.py         # Input/close pattern matching
    ├── prompt_builders.py  # Worker prompt construction
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

## Swarm Roles (Mandatory Multi-Agent Development)

ALL development uses the swarm pattern. No single-agent work allowed. Use `/swarm <task>` to execute.

```
┌──────────────────────────────────────────────────────────┐
│                   Coordinator (主控)                       │
│         Decompose → Dispatch → Apply → Verify             │
└──────┬──────┬───────────┬────────────┬──────────────────┘
       │      │           │            │
       ▼      ▼           ▼            ▼
  ┌────────┐ ┌──────────┐ ┌─────────┐ ┌───────────┐
  │   PM   │ │Developer │ │  E2E    │ │   Git     │
  │  Agent │ │ Agent(s) │ │ Tester  │ │  Manager  │
  └────────┘ └──────────┘ └─────────┘ └───────────┘
                    +
              ┌───────────┐
              │   Code    │
              │ Reviewer  │
              └───────────┘
```

| Role | Responsibility | Key Rule |
|------|---------------|----------|
| **PM Agent** | Requirements → user stories → acceptance criteria → CLAUDE.md updates | BLOCKING: runs first, every task |
| **Developer(s)** | TDD: failing tests → implementation (edit specs only) | Never writes files directly |
| **E2E Tester** | Business-level tests covering complete user flows | Runs twice: write tests + verify |
| **Code Reviewer** | Quality, security, thread safety, patterns | CRITICAL/HIGH must be fixed |
| **Git Manager** | Branch, atomic commits, PR, merge | Only after all tests pass |

## Development Workflow (MANDATORY Swarm Phases)

```
Phase 0: PM Agent        → requirements + acceptance criteria + CLAUDE.md updates
Phase 1: Developer(s)    → TDD edit specs (parallel per module)
         + E2E Tester    → business-level test edit specs (parallel)
Phase 2: Coordinator     → apply all edits + run tests
Phase 3: Code Reviewer   → quality/security review (parallel)
         + E2E Tester    → verify acceptance criteria (parallel)
Phase 4: Coordinator     → fix issues + re-verify
Phase 5: Git Manager     → branch + commit + PR
```

Skipping ANY phase is a workflow violation. See `.claude/commands/swarm.md` for full protocol.

### Key Constraints
- PM validates every request against the **AI Butler vision**
- PM **can modify this CLAUDE.md** to update project structure, skills, workflow
- E2E tests are **continuously maintained** — they evolve with requirements
- Git Manager is a **dedicated role**, not mixed into Coordinator
- All agent prompts include the project vision context

## Skills Reference

### Used by Swarm Roles
- **product-manager**: PM Agent's core skill
- **python-e2e-testing**: E2E Tester's core skill
- **python-reviewer**: Code Reviewer's core skill
- **supervisor-hub**: Supervisor-specific patterns for Developer Agents
- **tdd-workflow**: Developer Agents follow TDD discipline

### Situational (Coordinator decides)
- **python-patterns**: Pythonic idioms, type hints, PEP 8
- **security-review**: Input validation, secret management, subprocess security
- **api-design**: When adding/changing MCP tools
- **mcp-builder**: MCP server patterns
- **python-ci-setup**: CI/CD or pre-commit hooks

## Known Tech Debt (updated 2026-03-12)

| Issue | Severity | File | Status |
|-------|----------|------|--------|
| task_dispatcher.py still has re-export boilerplate | LOW | task_dispatcher.py (579) | Story 5 cleanup pending |
| ~41 pytest warnings (mostly ResourceWarnings) | HIGH | tests/*.py | Unfixed |
| Silent `pass` in except blocks | HIGH | supervisor/*.py | Unfixed |
| No CI/CD pipeline | HIGH | .github/ missing | Not started |
| No pre-commit hooks | HIGH | .pre-commit-config.yaml | Not started |
| Dependencies unpinned (bare >=) | HIGH | pyproject.toml | Unfixed |
| Coverage 71% (target 80%) | MEDIUM | -- | Below target |

## Quality Metrics

```
Tests: 619 passing, 0 failures, 41 warnings
Coverage: 71% (target: 80%)
Max file: 579 lines (task_dispatcher.py — includes re-exports)
Modules: 16 files in supervisor/ (avg ~300 lines)
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
3. Run `/swarm <task>` — all development uses swarm mode
4. After change: `pytest tests/ --tb=short` — 0 failures, no new warnings
5. After change: Code Reviewer agent confirms no CRITICAL/HIGH issues
6. Git Manager handles commits: one logical change per commit, conventional format
