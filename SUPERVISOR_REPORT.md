# Supervisor Hub 重构报告

## 1. 架构概览

### 消息流图

```
用户 (飞书)
  │
  ▼
┌──────────────────┐
│  FeishuGateway   │  WebSocket 长连接接收 + HTTP API 发送
│  (feishu_gateway) │  去重、过滤bot消息、@提及清理
└──────┬───────────┘
       │ _on_feishu_message()
       ▼
┌──────────────────────────────────────────────────────┐
│                  Supervisor (main.py)                  │
│                                                        │
│  Layer 1: /command 路由                                │
│    /status /tasks /gpu /sessions /help /close ...      │
│                                                        │
│  Layer 2: 智能消息路由 (_route_message)                │
│    ┌─────────────────────────────┐                     │
│    │ 1. 显式 task_id 匹配?       │                     │
│    │    → 直接追问 (follow_up)    │                     │
│    │ 2. Sonnet 分类 + 回复       │                     │
│    │    (build_route_prompt)      │                     │
│    └──────┬──────────────────────┘                     │
│           │                                            │
│    ┌──────┴──────────────────────┐                     │
│    │ action=reply                │→ 直接回复用户        │
│    │ action=dispatch             │→ 单任务下发          │
│    │ action=dispatch_multi       │→ 多任务并行拆解      │
│    │ action=follow_up            │→ 追问已完成任务      │
│    └─────────────────────────────┘                     │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐    ┌──────────────────────┐
│  ClaudeSession       │    │  TaskDispatcher       │
│  (claude_session.py) │    │  (task_dispatcher.py) │
│                      │    │                       │
│  route_message()     │    │  dispatch()           │
│  → sonnet 分类       │    │  → claude -p (opus)   │
│  call() / streaming  │    │  信号量控制并发         │
│  session持久化       │    │  streaming进度追踪     │
└──────────────────────┘    │  daemon自动重启        │
                            │  follow_up / resume   │
                            │  任务持久化到磁盘       │
                            └───────────┬───────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │  on_complete()   │
                              │  → _notify_task  │
                              │    _result()     │
                              │  → push to 飞书  │
                              └──────────────────┘

辅助组件:
┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐
│ ContainerMonitor│  │ SessionMonitor   │  │ Scheduler    │
│ CPU/MEM/GPU/Disk│  │ .jsonl session扫描│  │ 定时健康检查  │
│ 进程/端口监控    │  │ 任务进度提取      │  │ 会话摘要推送  │
└─────────────────┘  └──────────────────┘  │ 日报生成      │
                                            └──────────────┘
```

### 组件职责

| 模块 | 文件 | 行数 | 职责 |
|------|------|------|------|
| **Supervisor** | `main.py` | 729 | 核心调度器：消息入口、/command 处理、智能路由、结果通知、对话历史 |
| **RouterSkill** | `router_skill.py` | 187 | Sonnet 路由提示词构建：身份、规则、示例、任务上下文注入 |
| **ClaudeSession** | `claude_session.py` | 493 | Claude CLI 封装：流式/非流式调用、session持久化、JSON解析（含容错） |
| **TaskDispatcher** | `task_dispatcher.py` | 788 | 任务生命周期管理：创建、执行、并发控制、进度追踪、持久化、follow-up |
| **FeishuGateway** | `feishu_gateway.py` | 355 | 飞书双向网关：WebSocket接收、消息/图片/文件发送、PATCH更新 |
| **Scheduler** | `scheduler.py` | 234 | 定时任务：健康检查(5min)、会话摘要(15min)、日报(24h) |
| **ContainerMonitor** | `container_monitor.py` | 213 | 系统监控：CPU/内存/磁盘/GPU/进程/端口 |
| **SessionMonitor** | `session_monitor.py` | 410 | Claude会话扫描：JSONL解析、任务描述提取、活跃状态判断 |

---

## 2. Supervisor 5 大核心职责 — 实现状态

### 2.1 与用户交互 ✅ DONE

**实现方式：**
- `FeishuGateway.start_receiving()` (L156-177): WebSocket长连接接收用户消息
- `_handle_message()` (L91-132): 去重、bot过滤、@提及清理
- `_on_feishu_message()` in main.py (L615-644): 消息入口路由
- `/command` 系统 (L82-258): 12个命令 (/status, /tasks, /gpu, /sessions, /help, /stop, /skip, /close, /followup, /reply, /daemons)
- `push_message()` (L332-338): 主动推送结果到飞书
- `reply_message()` / `send_message()` / `update_message()`: 多种回复方式
- 消息已读事件追踪 (`_on_message_read`, L599-613)

**已知限制：**
- 仅支持文本消息作为输入，图片/文件消息不会被解析为指令
- `@_user_1` 硬编码为bot mention标识，不够通用

### 2.2 理解用户意图 ✅ DONE

**实现方式：**
- `build_route_prompt()` in router_skill.py (L134-186): 构建包含身份、规则、示例、任务上下文、对话历史的完整 prompt
- `route_message()` in claude_session.py (L285-365): 调用 sonnet 分类 + 生成回复
- 4种动作分类: reply / dispatch / dispatch_multi / follow_up
- 显式 task_id 匹配 (`_extract_task_id_from_text`, L264-288): 8位hex前缀直接路由
- 任务上下文注入 (`_get_tasks_context`, L290-326): 活跃/等待关闭任务信息传给sonnet
- 对话历史注入 (`_get_history_text`, L403-411): 最近20条消息上下文
- 健壮的JSON解析容错 (L367-468): 标准解析 → regex提取 → 纯文本启发式 → 安全fallback

**已知限制：**
- sonnet 路由延迟约3-5秒，对简单问候也会走完整路由
- `_INPUT_PHRASES` 启发式 (L28-34 in task_dispatcher.py) 仅覆盖英文短语，中文场景覆盖不足

### 2.3 编排/下发任务 ✅ DONE

**实现方式：**
- `dispatch()` in task_dispatcher.py (L355-397): 创建Task、加入后台执行队列
- `_oneshot_worker()` (L289-293): 信号量限制并发 (默认 MAX_WORKERS=3)
- `_daemon_worker()` (L296-323): 自动重启，最多3次重试
- `_handle_dispatch_multi()` in main.py (L470-502): 拆解为并行子任务
- `_build_worker_prompt()` (L413-439): 为worker构建enriched prompt（含对话历史和任务描述）
- Streaming进度追踪 (`_run_claude`, L184-284): 实时解析tool_use事件，记录step
- `on_complete` 回调机制 (L328-349): 任务完成后通知

**已知限制：**
- 子任务之间无依赖关系管理（全部并行，无DAG编排）
- 无优先级队列机制

### 2.4 监控 Worker 状态 ⚠️ PARTIAL

**实现方式：**
- Task dataclass (L80-98): 记录 status, current_step, steps_completed, started_at, finished_at, error
- `_set_status()` (L174-178): 每次状态变更记录日志 + 持久化到磁盘
- `_format_task()` (L708-744): 格式化任务显示（状态、耗时、进度、当前步骤）
- `/tasks` 和 `/status` 命令实时查询
- `_get_tasks_context()` (L290-326): 将任务进度信息注入路由 prompt，使 sonnet 能回答进度查询
- `SessionMonitor` (session_monitor.py): 扫描 .jsonl 文件追踪所有 Claude 会话
- `Scheduler._run_health_check()` (L106-157): 系统资源阈值告警

**已知限制：**
- **无实时进度推送**: 任务执行中不主动推送进度到飞书，用户必须主动查询
- **无超时机制**: task_dispatcher 中的 `_run_claude` 没有全局超时，opus 任务可能无限运行
- **无心跳检测**: 不检测 worker 进程是否假死（stdout无输出但进程未退出）

### 2.5 返回 Worker 结果 ✅ DONE

**实现方式：**
- `_notify_task_result()` in main.py (L553-593): 任务完成后推送结果到飞书
  - `awaiting_closure` → "任务完成" + 结果文本 + 追问/关闭提示
  - `waiting_for_input` → "需要输入" + /reply 提示
  - `failed` → "任务失败" + 错误信息
- 结果截断保护 (L567-568): 超过3000字符截断
- 对话历史记录 (L588): 结果摘要记入对话历史
- `follow_up_async()` (L507-564): 在现有session上追问，保持上下文
- `resume_task()` (L400-457): 恢复等待输入的任务

**已知限制：**
- 结果推送可能因飞书API限流失败，仅记日志无重试
- 长结果截断到3000字符，没有提供"查看完整结果"机制

---

## 3. 各模块变更总结

### Module A: FeishuGateway (feishu_gateway.py)
- **双向通信**: WebSocket接收 + HTTP API发送
- **消息去重**: TTL 60s 的消息ID去重
- **Bot过滤**: 防止自我消息循环
- **多媒体支持**: 图片/文件上传和发送
- **流式更新**: PATCH API支持消息内容更新
- **已读事件**: 注册 `p2_im_message_message_read_v1` 事件

### Module B: ClaudeSession (claude_session.py)
- **统一路由**: `route_message()` 用 sonnet 分类 + 生成回复，一次调用完成
- **健壮解析**: 三层JSON解析容错 (标准 → regex → 启发式)
- **Session持久化**: 文件 `~/.supervisor/main_session_id`
- **Session过期处理**: 自动重试一次（L264-267）
- **Markdown剥离**: 处理 sonnet 输出被 ` ```json ` 包裹的情况

### Module C: TaskDispatcher (task_dispatcher.py)
- **完整任务生命周期**: pending → running → awaiting_closure / waiting_for_input / failed → completed
- **Streaming进度追踪**: 解析 stream-json 中的 tool_use 事件
- **并发控制**: 双信号量（worker 3个、daemon 2个）
- **Daemon自动重启**: 失败后最多重试3次
- **任务持久化**: JSON文件 `/tmp/supervisor-tasks.json`，重启恢复
- **Follow-up**: 在已有session上继续对话
- **Description**: 支持显式描述或自动从prompt生成

### Module D: Supervisor Main (main.py) + RouterSkill + Scheduler
- **两层路由**: /command → sonnet智能路由
- **4种动作**: reply / dispatch / dispatch_multi / follow_up
- **任务上下文感知**: sonnet路由时注入活跃任务和等待关闭任务信息
- **对话历史**: 最近20条消息记录，注入worker prompt和route prompt
- **定时调度**: 健康检查(5min)、会话摘要(15min)、日报(24h)
- **告警去重**: 相同告警不重复推送

---

## 4. Code Review 发现

### CRITICAL 问题

1. **task_dispatcher.py 全局可变状态** (L112-116)
   - `_tasks`, `_background_handles`, `_worker_semaphore`, `_daemon_semaphore` 均为模块级全局变量
   - 多线程环境下无锁保护（`_on_feishu_message` 在 WebSocket 线程，`_route_message` 在 asyncio 线程）
   - `_save_tasks()` 使用 tmp+rename 原子写入，但 `_tasks` dict 的读写本身不是线程安全的

2. **`bypassPermissions` 模式** (L24 in claude_session.py, L51 in task_dispatcher.py)
   - Worker 以 `--permission-mode bypassPermissions` 运行，可执行任意命令
   - 用户输入直接作为 prompt 传递给 worker，无沙箱限制
   - 恶意输入可能导致任意代码执行

### HIGH 问题

3. **Worker 无超时保护** (task_dispatcher.py `_run_claude`)
   - `_run_claude()` 没有全局超时，opus 会话可能运行数小时
   - `asyncio.wait_for` 仅用于 ClaudeSession.call()（10分钟超时），dispatcher 中未使用
   - 建议: 添加可配置超时 (如 `SUPERVISOR_TASK_TIMEOUT=1800`)

4. **`_read_messages` 集合无序截断** (main.py L607-609)
   - 使用 `set.pop()` 随机移除元素，可能移除最近的消息ID
   - 建议: 改用 `collections.OrderedDict` 或带时间戳的结构

5. **feishu_gateway.py 文件句柄泄漏** (L273, L302)
   - `open(image_path, "rb")` 和 `open(file_path, "rb")` 未使用 `with` 语句
   - 建议: 使用 context manager 或手动关闭

6. **task_dispatcher.py 模块导入时执行** (L157)
   - `_load_tasks()` 在模块 import 时执行，可能影响测试隔离
   - 测试中通过 `_reset()` 规避，但增加了脆弱性

### MEDIUM 问题

7. **main.py 接近 800 行上限** (729行)
   - 建议: 将 `/command` 处理器提取到 `commands.py`，将 action handler 提取到 `actions.py`

8. **task_dispatcher.py 已达 788 行**
   - 建议: 将 formatting 函数 (L674-773) 提取到独立模块

9. **`_looks_like_needs_input` 启发式过于简单** (task_dispatcher.py L28-34)
   - 仅检测英文短语 (please, which, should i, confirm)
   - 中文询问（"请问", "你想要", "选择哪个"）全部漏检
   - 建议: 增加中文短语或使用 sonnet 判断

10. **Scheduler 没有 jitter/offset** (scheduler.py)
    - 三个定时任务同时启动，可能在同一时刻触发
    - 建议: 添加随机偏移避免请求风暴

---

## 5. 测试覆盖

### 测试统计

| 测试文件 | 测试用例数 | 覆盖模块 |
|----------|-----------|----------|
| `test_supervisor_main.py` | 48 | main.py (command/route/notify/history) |
| `test_claude_session.py` | 50 | claude_session.py (build_cmd/parse/route/regex) |
| `test_task_dispatcher.py` | 47 | task_dispatcher.py (dispatch/daemon/resume/format) |
| `test_router_skill.py` | 22 | router_skill.py (prompt building/context) |
| `test_feishu_gateway.py` | 18 | feishu_gateway.py (dedup/init/send) |
| `test_scheduler.py` | 15 | scheduler.py (health/digest/daily/lifecycle) |
| `test_session_monitor.py` | 29 | session_monitor.py |
| `test_container_monitor.py` | 14 | container_monitor.py |
| `test_utils.py` | 11 | utils模块 |
| `test_drive_security.py` | 8 | drive安全相关 |
| **总计** | **262** | **所有 supervisor 模块** |

### 关键测试覆盖亮点
- **路由分类全路径**: reply / dispatch / dispatch_multi / follow_up 各有测试
- **JSON解析容错**: 中文未转义引号、markdown包裹、空结果、超时 全部覆盖
- **任务生命周期**: pending→running→awaiting_closure→completed 完整链路
- **Daemon重试**: 失败自动重启+最大重试次数
- **并发控制**: 信号量限制并发数
- **边界情况**: 对话历史上限、结果截断、未知命令、dispatcher不可用
- **持久化恢复**: _load_tasks null字段处理、孤立.tmp恢复、running→failed转换

### 覆盖缺口
- **集成测试**: 无端到端集成测试（需要真实飞书环境）
- **Scheduler 定时触发**: 仅测试单次执行逻辑，未测试 asyncio 定时循环
- **WebSocket 断线重连**: `FeishuGateway.start_receiving()` 未被测试
- **`_handle_dispatch_multi` 子任务全部完成后的聚合**: 无测试

---

## 6. 待办事项（优先级排列）

### P0 — 必须立即修复

| # | 事项 | 原因 | 影响文件 |
|---|------|------|----------|
| 1 | Worker 任务添加超时保护 | opus 任务可能无限运行，耗尽资源 | `task_dispatcher.py` |
| 2 | 全局可变状态添加锁保护 | 多线程读写 `_tasks` dict 可能导致数据损坏 | `task_dispatcher.py` |

### P1 — 下一迭代完成

| # | 事项 | 原因 | 影响文件 |
|---|------|------|----------|
| 3 | 中文 `_looks_like_needs_input` 扩展 | 中文用户场景下 waiting_for_input 判断几乎失效 | `task_dispatcher.py` |
| 4 | 修复 `_read_messages` 无序截断 | 可能丢失最近消息的已读状态 | `main.py` |
| 5 | 修复文件句柄泄漏 | 高频调用可能耗尽文件描述符 | `feishu_gateway.py` |
| 6 | 子任务聚合通知 | 用户无法知道 dispatch_multi 的所有子任务是否都完成 | `main.py` |
| 7 | 任务执行中进度推送 | 长任务（>30s）用户无反馈，体验差 | `main.py`, `task_dispatcher.py` |

### P2 — 后续优化

| # | 事项 | 原因 | 影响文件 |
|---|------|------|----------|
| 8 | 拆分 main.py (commands + actions) | 已接近800行上限 | `main.py` → 新文件 |
| 9 | 拆分 task_dispatcher.py formatting | 已达788行 | `task_dispatcher.py` → 新文件 |
| 10 | Scheduler 添加 jitter | 避免定时任务风暴 | `scheduler.py` |
| 11 | 结果推送失败重试机制 | 飞书API限流时结果丢失 | `main.py` |
| 12 | 简单问候快速通道 | 避免"你好"也走sonnet路由 (3-5s延迟) | `main.py` |
| 13 | 添加端到端集成测试 | 验证完整消息流 | `tests/` |
| 14 | 任务优先级队列 | 当前FIFO，紧急任务无法插队 | `task_dispatcher.py` |

---

## 7. 风险评估

### 生产环境风险矩阵

| 风险 | 概率 | 影响 | 当前缓解 | 建议 |
|------|------|------|----------|------|
| **Worker 无限运行** | 高 | 高 — 资源耗尽 | 无 | P0: 添加超时 |
| **`_tasks` 竞态条件** | 中 | 高 — 数据损坏 | 无 | P0: 添加锁 |
| **飞书 WebSocket 断线** | 中 | 高 — 完全失联 | lark SDK 内部重连 | 监控+告警 |
| **Sonnet 路由错误** | 中 | 中 — 该reply的被dispatch | dispatch作为安全fallback | 可接受 |
| **任务持久化丢失** | 低 | 中 — /tmp 重启清空 | `_load_tasks()` 恢复 | 改用非/tmp目录 |
| **Claude CLI 不可用** | 低 | 高 — 所有任务失败 | 错误消息反馈 | 启动时检测 |
| **飞书 API 限流** | 中 | 低 — 消息延迟 | 仅日志记录 | 添加重试+退避 |
| **对话历史内存泄漏** | 低 | 低 — 上限20条 | MAX_HISTORY=20 | 已缓解 |

### 需要监控的关键指标

1. **Worker 并发数**: 当前活跃任务数 vs MAX_WORKERS(3)
2. **Sonnet 路由延迟**: route_message() 调用耗时
3. **任务完成率**: awaiting_closure / (awaiting_closure + failed) 比率
4. **飞书推送成功率**: push_message() 成功/失败比
5. **系统资源**: CPU/MEM/GPU (已有 Scheduler 监控)
6. **Task 持久化文件大小**: `/tmp/supervisor-tasks.json` 是否无限增长（仅24h后清理 completed/cancelled）

### 关键建议

**架构整体评价**: Supervisor Hub 的分层架构（Gateway → Router → Dispatcher → Worker）清晰合理，sonnet分类+opus执行的双模型策略成本效益好。核心功能（5大职责中的4.5个）已完整实现，测试覆盖262个用例。主要风险集中在运维侧（超时、线程安全、断线恢复），而非功能缺失。建议优先处理P0的超时和线程安全问题后即可投入生产使用。

---

## 8. 本轮已修复问题

以下问题在本次重构中已修复并通过测试验证：

| 优先级 | 问题 | 修复方式 | 测试 |
|--------|------|----------|------|
| HIGH | 显式 task ID follow-up 后未记录对话历史 | `main.py:347` 添加 `_record_message` | ✅ |
| HIGH | `_load_tasks` 中 `finished_at=None` 导致崩溃 | None float 字段强制转为 `0.0` | ✅ 3 new tests |
| HIGH | `dispatch_multi` 子任务缺少 `description` | 传递 `description=sub_prompt[:80]` | ✅ |
| MEDIUM | 中断保存导致孤立 `.tmp` 文件 | `_load_tasks` 启动时检测并恢复 | ✅ |
| MEDIUM | `message_read_v1` 事件未注册导致 ERROR 日志 | 注册真实事件处理器 | ✅ |
| MEDIUM | Sonnet 胡编任务状态 | 反捏造规则 + 丰富任务上下文注入 | ✅ |
| MEDIUM | 任务重启后消失 | 任务持久化到 JSON 文件 | ✅ |
| LOW | 任务描述显示为 worker preamble | `dispatch()` 增加 `description` 参数 | ✅ 3 tests |

---

**报告涉及的关键文件路径:**

- `/workspace/feishu-mcp-server/src/supervisor/main.py` (729行，核心调度)
- `/workspace/feishu-mcp-server/src/supervisor/router_skill.py` (187行，路由提示词)
- `/workspace/feishu-mcp-server/src/supervisor/claude_session.py` (493行，Claude CLI封装)
- `/workspace/feishu-mcp-server/src/supervisor/task_dispatcher.py` (788行，任务管理)
- `/workspace/feishu-mcp-server/src/supervisor/feishu_gateway.py` (355行，飞书网关)
- `/workspace/feishu-mcp-server/src/supervisor/scheduler.py` (234行，定时调度)
- `/workspace/feishu-mcp-server/src/supervisor/container_monitor.py` (213行，系统监控)
- `/workspace/feishu-mcp-server/src/supervisor/session_monitor.py` (410行，会话监控)
- `/workspace/feishu-mcp-server/tests/test_supervisor_main.py` (48个测试)
- `/workspace/feishu-mcp-server/tests/test_claude_session.py` (50个测试)
- `/workspace/feishu-mcp-server/tests/test_task_dispatcher.py` (47个测试)
- `/workspace/feishu-mcp-server/tests/test_router_skill.py` (22个测试)
