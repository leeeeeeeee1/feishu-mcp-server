# Supervisor Hub 修复与增强 — 开发计划

> **PM/Researcher 输出** | 日期: 2026-03-12
> 基于 SUPERVISOR_REPORT.md 分析、代码审查、运行日志和用户反馈

---

## 一、问题清单总览

### 1.1 用户反馈的关键问题

| # | 问题 | 严重性 | 来源 |
|---|------|--------|------|
| **U1** | 智能关闭失败但返回成功状态 — Sonnet 返回 action="reply" 并生成"已关闭"文本，但实际未执行 close_task() | **CRITICAL** | 运行日志 |
| **U2** | 大量硬编码模式匹配 — `_CLOSE_PHRASES`, `_CLOSE_INTENT_PATTERNS`, `_INPUT_PHRASES`, `_ACTION_VERBS` 等全部是写死的正则/短语 | **HIGH** | 用户明确要求 |
| **U3** | Sonnet 路由理解准确性不足 — 需要通过更好的 prompt 设计、上下文信息、结构化输出等方式提升 | **HIGH** | 用户明确要求 |

### 1.2 SUPERVISOR_REPORT.md 中的 P0-P2 问题

| # | 问题 | 严重性 | 影响文件 |
|---|------|--------|----------|
| **R1** | Worker 无超时保护 — opus 任务可能无限运行 | **P0-CRITICAL** | task_dispatcher.py |
| **R2** | 全局可变状态 `_tasks` 无锁保护 — 多线程读写数据损坏风险 | **P0-CRITICAL** | task_dispatcher.py |
| **R3** | `_read_messages` 无序截断 — 随机丢失最近已读状态 | **P1-HIGH** | main.py |
| **R4** | feishu_gateway.py 文件句柄泄漏 — upload_image/upload_file 未用 with | **P1-HIGH** | feishu_gateway.py |
| **R5** | dispatch_multi 子任务无聚合通知 — 用户不知道所有子任务是否完成 | **P1-HIGH** | main.py |
| **R6** | 长任务无实时进度推送 — >30s 任务用户无反馈 | **P1-HIGH** | main.py, task_dispatcher.py |
| **R7** | main.py 接近 800 行上限 | **P2-MEDIUM** | main.py |
| **R8** | task_dispatcher.py 已达 788→1234 行 | **P2-MEDIUM** | task_dispatcher.py |
| **R9** | Scheduler 无 jitter — 定时任务可能同时触发 | **P2-MEDIUM** | scheduler.py |
| **R10** | 结果推送无重试 — 飞书 API 限流时结果丢失 | **P2-MEDIUM** | main.py |

### 1.3 代码审查发现的额外问题

| # | 问题 | 严重性 | 影响文件 |
|---|------|--------|----------|
| **C1** | `_load_tasks()` 模块导入时执行 — 影响测试隔离 | **MEDIUM** | task_dispatcher.py:358 |
| **C2** | `_looks_like_needs_input()` 仅英文 — 中文场景完全失效 | **MEDIUM** | task_dispatcher.py:36-41 |
| **C3** | `close_task()` 静默吞掉非 awaiting_closure 状态的错误 | **MEDIUM** | task_dispatcher.py:895-910 |
| **C4** | `_message_task_map` 容量上限逻辑有 off-by-one 风险 | **LOW** | main.py:777-780 |

---

## 二、根因分析 — 三大核心问题

### 2.1 智能关闭失败但返回成功 (U1)

**根因链:**

```
用户: "这个可以关了" (指代某个 awaiting_closure 的任务)
    ↓
_route_message() Step 3: Sonnet 路由
    ↓
Sonnet 返回: {"action": "reply", "text": "好的，任务已关闭"}  ← BUG: 错误分类
    ↓
main.py:455-460: 处理 action="reply"
    → self.gateway.reply_message(message_id, "好的，任务已关闭")
    → 用户看到 "好的，任务已关闭"
    → 但 close_task() 从未被调用！
    → 任务仍然是 awaiting_closure 状态
```

**另一个路径:**
```
Sonnet 返回: {"action": "follow_up", "task_id": "aabb1122", "text": "关闭任务"}
    ↓
_handle_sonnet_follow_up(): 检测到 _contains_close_intent("这个可以关了")
    → 走 smart close fallback → _handle_sonnet_close()
    → 但 task_id_prefix 可能匹配失败（Sonnet 给了错误的 task_id）
    → 返回 "未找到匹配的待关闭任务"
    → 但用户预期是成功的
```

**修复方向:**
1. 在 action="reply" 处理后，增加二次验证：如果 reply 文本包含关闭相关语义且有待关闭任务，执行实际关闭
2. 改进 Sonnet prompt，明确区分"回复关闭确认"和"执行关闭操作"
3. 增加关闭操作的结果验证，确保状态确实变更

### 2.2 硬编码模式匹配 (U2)

**当前硬编码清单:**

| 位置 | 变量/函数 | 硬编码内容 | 行数 |
|------|----------|-----------|------|
| task_dispatcher.py:33 | `_INPUT_PHRASES` | 4个英文短语 | 仅英文 |
| task_dispatcher.py:45-48 | `_CLOSE_PHRASES` | 13个中英文短语 | 固定集合 |
| task_dispatcher.py:57-67 | `_CLOSE_FALSE_POSITIVES` | 15个技术名词正则 | 固定正则 |
| task_dispatcher.py:69-79 | `_CLOSE_INTENT_PATTERNS` | 9个关闭意图正则 | 固定正则 |
| task_dispatcher.py:82-92 | `_contains_close_intent()` | 组合硬编码检测 | 依赖上述 |
| task_dispatcher.py:95-113 | `_looks_like_close()` | 精确匹配+长度限制 | 依赖上述 |
| claude_session.py:338-342 | `_ACTION_VERBS` | 20个动作词正则 | 固定正则 |

**用户要求:** 不要任何写死的模式匹配。通过 prompt 设计、上下文、结构化输出提升 Sonnet 理解准确性。

**修复方向:**
1. 删除所有硬编码模式匹配函数和变量
2. 将关闭意图检测、输入需求检测全部交给 Sonnet 判断
3. 在路由结果中增加结构化字段（如 `close_intent: bool`, `needs_input: bool`）
4. 通过 few-shot examples 和更精确的 prompt 引导 Sonnet 准确分类
5. Worker 完成后的状态判断（需要输入 vs 完成）也改用 Sonnet 评估

### 2.3 Sonnet 路由理解准确性 (U3)

**当前问题:**
- Router prompt 过长（~3000 tokens system prompt），信噪比低
- 规则描述用自然语言，边界模糊
- 缺乏负面示例（"这个不是 close"）
- 无结构化输出约束（JSON schema）
- close/follow_up/reply 边界在多任务场景下容易混淆

**修复方向:**
1. 重构 router_skill.py prompt — 精简规则、增加对比示例
2. 添加 JSON Schema 约束（Anthropic API 支持 tool_use 结构化输出）
3. 增加 close vs follow_up vs reply 的边界案例示例
4. 将 worker 完成后的状态判断也用 Sonnet（替代 `_looks_like_needs_input`）
5. 结果验证层：action 与上下文一致性校验

---

## 三、模块化开发方案

### 模块 A: Sonnet 智能增强 — 消除硬编码模式匹配

**负责 Agent:** `sonnet-intelligence`
**优先级:** CRITICAL (U1 + U2 + U3 的核心修复)
**影响文件:** `router_skill.py`, `claude_session.py`, `task_dispatcher.py`, `main.py`
**估计复杂度:** 高

#### A1: 重构路由 Prompt (router_skill.py)

**当前问题:** prompt 信噪比低，close/follow_up 边界模糊

**修改方案:**
```
1. 精简 ROUTING_RULES:
   - 删除冗余描述，用表格代替长段落
   - 明确 close 的唯一判断标准：用户表达了"结束/关闭/不需要了"的意图
   - 增加反面示例："关闭连接" ≠ close task

2. 增强 ROUTING_EXAMPLES:
   - 增加 close vs follow_up 的对比示例组（至少5对）
   - 增加 close vs reply 的对比示例组
   - 增加多任务场景的匹配示例
   - 增加"用户说了关闭但不是指任务"的负面示例

3. 结构化输出:
   - 在 system prompt 中定义 JSON Schema
   - 利用 Anthropic API 的 tool_use 模式强制结构化输出
   - 增加字段: needs_input (bool), close_intent (bool)
```

**关键修改点:**
- `router_skill.py`: 重写 `ROUTING_RULES`, `ROUTING_EXAMPLES`, `_ROUTE_SYSTEM`, `_ROUTE_USER`
- `router_skill.py`: 新增 `ROUTE_OUTPUT_SCHEMA` 定义
- `router_skill.py`: 新增 close/follow_up 边界对比示例

#### A2: 结构化输出模式 (claude_session.py)

**当前问题:** Sonnet 输出自由文本 JSON，解析需要3层容错

**修改方案:**
```
1. API 路由模式改用 tool_use:
   - 定义 route_decision tool
   - 输入参数即为 action + 各字段
   - 利用 Anthropic API 的 tool_use 强制结构化，消除 JSON 解析问题

2. CLI 路由模式保留现有 JSON 解析:
   - 作为 fallback，保留 _try_json_parse + _try_regex_extract
   - 但不再依赖 _ACTION_VERBS 等硬编码模式

3. 删除 _ACTION_VERBS 硬编码
```

**关键修改点:**
- `claude_session.py`: `_route_via_api()` 改用 tool_use 模式
- `claude_session.py`: 删除 `_ACTION_VERBS` 正则
- `claude_session.py`: 重写 plain text fallback 逻辑

#### A3: 消除硬编码检测函数 (task_dispatcher.py + main.py)

**当前问题:** `_looks_like_close`, `_contains_close_intent`, `_looks_like_needs_input` 全部硬编码

**修改方案:**
```
1. 删除以下硬编码:
   - _INPUT_PHRASES
   - _CLOSE_PHRASES / _CLOSE_PHRASES_SET
   - _CLOSE_FALSE_POSITIVES
   - _CLOSE_INTENT_PATTERNS
   - _contains_close_intent()
   - _looks_like_close()
   - _looks_like_needs_input()

2. 替代方案:
   a) Worker 完成后的状态判断:
      - 新增 _classify_completion(result_text) -> "completed" | "needs_input"
      - 调用 Sonnet API 快速判断（<1s，haiku 也可以）
      - Prompt: "Given this worker output, does it require user input? Reply JSON {needs_input: bool}"

   b) 用户消息的 close 意图:
      - 完全交给路由层 Sonnet 判断
      - 在 _route_message Step 0 (reply-based follow-up) 中:
        - 不再用 _looks_like_close 做本地判断
        - 改为调用一次轻量 Sonnet 分类
        - 或直接走 Step 3 的 Sonnet 路由（已有 awaiting task 上下文）

   c) Smart close fallback:
      - _handle_sonnet_follow_up 中的 _contains_close_intent 调用删除
      - 如果 Sonnet 说 follow_up，就信任它是 follow_up
      - 如果 Sonnet 分类错误，用户可以再说一次或用 /close 命令
```

**关键修改点:**
- `task_dispatcher.py`: 删除 L33-113 的所有硬编码模式匹配
- `task_dispatcher.py`: 新增 `classify_completion()` 异步函数
- `main.py:397-410`: 删除 `_looks_like_close` 调用，改用 Sonnet 或直接走 _route_message
- `main.py:647-655`: 删除 `_contains_close_intent` 调用

#### A4: 关闭操作可靠性增强 (main.py)

**当前问题:** Sonnet 可能返回 action="reply" + text="已关闭"，但未执行关闭

**修改方案:**
```
1. 在 action="reply" 处理后增加验证层:
   - 如果有 awaiting_closure 任务，且 reply 文本中包含"关闭/closed"语义
   - 记录 WARNING 日志（但不自动关闭 — 避免误操作）
   - 在回复中附加提示："如需关闭任务，请使用 /close <id>"

2. 更好的方案（推荐）:
   - 在 Sonnet prompt 中强调：
     "你只能通过 action='close' 来关闭任务。
      action='reply' 的 text 中绝不能包含'已关闭/任务关闭'等表述。
      如果你在 reply 中说了'已关闭'但没有用 close action，这是错误的。"
   - 增加负面示例

3. 结果通知中增加状态确认:
   - close 操作后发送确认消息，包含任务最终状态
   - 不依赖 Sonnet 的回复文本，而是从 close_task() 的返回值构建消息
```

**关键修改点:**
- `router_skill.py`: 增加反关闭幻觉规则和负面示例
- `main.py:455-460`: action="reply" 处理后增加 awaiting task 检查
- `main.py:660-711`: `_handle_sonnet_close()` 增加状态确认

---

### 模块 B: 线程安全与超时保护

**负责 Agent:** `thread-safety`
**优先级:** P0-CRITICAL (R1 + R2)
**影响文件:** `task_dispatcher.py`
**估计复杂度:** 中

#### B1: 全局状态加锁 (R2)

**修改方案:**
```python
import threading

_tasks_lock = threading.Lock()

# 所有读写 _tasks 的操作用 lock 保护:
def _set_status(task, new_status):
    with _tasks_lock:
        old = task.status
        task.status = new_status
        _save_tasks()
    logger.info(...)

def get_task(task_id):
    with _tasks_lock:
        task = _tasks.get(task_id)
    ...

def list_tasks():
    with _tasks_lock:
        return list(_tasks.values())
```

**关键修改点:**
- `task_dispatcher.py`: 新增 `_tasks_lock = threading.Lock()`
- `task_dispatcher.py`: 所有读写 `_tasks` 的函数加 `with _tasks_lock:`
- 受影响函数: `_set_status`, `_save_tasks`, `_load_tasks`, `dispatch`, `get_task`, `list_tasks`, `close_task`, `close_tasks`, `cancel_task`, `stop_daemon`, `get_awaiting_closure`, `list_daemons`, `list_interrupted`, `get_review_pending`

#### B2: Worker 超时保护 (R1)

**修改方案:**
```python
SUPERVISOR_TASK_TIMEOUT = int(os.environ.get("SUPERVISOR_TASK_TIMEOUT", "1800"))  # 30min default

async def _run_claude(task: Task) -> None:
    try:
        await asyncio.wait_for(
            _run_claude_inner(task),
            timeout=SUPERVISOR_TASK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        task.error = f"Task timed out after {SUPERVISOR_TASK_TIMEOUT}s"
        task.finished_at = time.time()
        _set_status(task, "failed")
        # Kill the subprocess if still running
        ...
```

**关键修改点:**
- `task_dispatcher.py`: 新增 `SUPERVISOR_TASK_TIMEOUT` 配置
- `task_dispatcher.py`: `_run_claude()` 用 `asyncio.wait_for` 包裹
- `task_dispatcher.py`: 超时后清理子进程

#### B3: _load_tasks() 延迟加载 (C1)

**修改方案:**
```python
# 删除 L358: _load_tasks()
# 改为在 dispatch() 首次调用时加载
_loaded = False

def _ensure_loaded():
    global _loaded
    if not _loaded:
        _load_tasks()
        _loaded = True
```

**关键修改点:**
- `task_dispatcher.py:358`: 删除模块级 `_load_tasks()` 调用
- `task_dispatcher.py`: 新增 `_ensure_loaded()` 函数
- `task_dispatcher.py`: 在 `dispatch()`, `get_task()`, `list_tasks()` 等入口调用

---

### 模块 C: 网关与监控增强

**负责 Agent:** `gateway-monitor`
**优先级:** P1-HIGH (R3 + R4 + R5 + R6)
**影响文件:** `main.py`, `feishu_gateway.py`, `task_dispatcher.py`
**估计复杂度:** 中

#### C1: 修复 _read_messages 截断 (R3)

**当前问题:** main.py:794-798 使用 dict 但之前报告说用 set.pop()

**实际代码检查:** 当前代码已改为 `dict[str, float]`，截断逻辑正确（保留最新500条）。
**但仍有问题:** `keys[:len(keys) - 500]` 在 len(keys) == 501 时删除 1 个最老的，正确。

**修改方案:** 确认已修复，无需额外修改。如果性能敏感可改用 `collections.OrderedDict`。

#### C2: 修复文件句柄泄漏 (R4)

**修改方案:**
```python
# feishu_gateway.py upload_image():
# Before:
#   file=open(image_path, "rb")
# After:
with open(image_path, "rb") as f:
    req = CreateImageRequest.builder()...file(f)...build()
```

**关键修改点:**
- `feishu_gateway.py`: `upload_image()` — 用 `with` 包裹 `open()`
- `feishu_gateway.py`: `upload_file()` — 同上

#### C3: 子任务聚合通知 (R5)

**修改方案:**
```python
# main.py: 在 _handle_dispatch_multi 中跟踪子任务
# 新增: _multi_task_tracker: dict[str, MultiTaskGroup]

@dataclass
class MultiTaskGroup:
    parent_message_id: str
    chat_id: str
    description: str
    task_ids: list[str]
    completed_count: int = 0
    total_count: int = 0

# 在 on_complete 回调中:
def on_complete(task, _cid=chat_id, _group_id=group_id):
    self._notify_task_result(task, _cid)
    group = self._multi_task_groups.get(_group_id)
    if group:
        group.completed_count += 1
        if group.completed_count == group.total_count:
            self.gateway.send_message(
                _cid,
                f"📊 所有子任务已完成 ({group.total_count}/{group.total_count})\n{group.description}"
            )
```

**关键修改点:**
- `main.py`: 新增 `MultiTaskGroup` dataclass
- `main.py`: `__init__` 新增 `_multi_task_groups: dict`
- `main.py`: `_handle_dispatch_multi()` 创建 group 并跟踪

#### C4: 长任务进度推送 (R6)

**修改方案:**
```python
# task_dispatcher.py: streaming 模式下每 N 步推送进度
# 在 _run_claude_streaming 中:

async def _run_claude_streaming(task, env, on_progress=None):
    ...
    step_count = 0
    last_push = time.time()
    PUSH_INTERVAL = 30  # seconds

    for block in msg.get("content", []):
        if block.get("type") == "tool_use":
            step_count += 1
            if on_progress and (time.time() - last_push > PUSH_INTERVAL):
                on_progress(task, step_count)
                last_push = time.time()
```

**关键修改点:**
- `task_dispatcher.py`: `dispatch()` 增加 `on_progress` 回调参数
- `task_dispatcher.py`: `_run_claude_streaming()` 定期调用 on_progress
- `main.py`: `_handle_dispatch()` 传入 progress 回调，推送到飞书

---

### 模块 D: 代码重构与组织

**负责 Agent:** `code-restructure`
**优先级:** P2-MEDIUM (R7 + R8 + R9)
**影响文件:** `main.py`, `task_dispatcher.py`, `scheduler.py`
**估计复杂度:** 中

#### D1: 拆分 main.py (R7)

**方案:**
```
main.py (920行) → 拆分为:
├── main.py         (~300行) — Supervisor 核心: __init__, _run_async, run, _on_feishu_message
├── commands.py     (~200行) — /command 处理器: _cmd_status, _cmd_tasks, _cmd_close 等
├── actions.py      (~300行) — action 处理器: _handle_dispatch, _handle_follow_up, _handle_sonnet_close 等
└── notification.py (~100行) — _notify_task_result, _record_message, _get_history_text
```

#### D2: 拆分 task_dispatcher.py (R8)

**方案:**
```
task_dispatcher.py (1234行) → 拆分为:
├── task_dispatcher.py  (~600行) — 核心: Task, dispatch, _run_claude, resume, follow_up, close
├── task_formatter.py   (~200行) — 格式化: _format_task, get_tasks_text, get_daemons_text
├── task_persistence.py (~200行) — 持久化: _save_tasks, _load_tasks, _save_checkpoint, _load_checkpoint
└── task_recovery.py    (~150行) — 恢复: recover_task, list_interrupted
```

#### D3: Scheduler jitter (R9)

**修改方案:**
```python
import random

async def _schedule_loop(self, name, interval, func):
    # 随机偏移 0-60s 避免风暴
    await asyncio.sleep(random.uniform(0, 60))
    while not self._stop_event.is_set():
        await func()
        await asyncio.sleep(interval)
```

---

### 模块 E: 测试与验证

**负责 Agent:** `tester`
**优先级:** BLOCKING GATE (每个模块的测试)
**影响文件:** `tests/` 目录
**估计复杂度:** 高

#### E1: 模块 A 测试 — Sonnet 智能增强

```
新增/修改:
- test_router_skill.py: 新增 close vs follow_up 边界测试 (≥10 cases)
- test_claude_session.py: 新增 tool_use 结构化输出测试
- test_claude_session.py: 删除 _ACTION_VERBS 相关测试，新增 Sonnet fallback 测试
- test_task_dispatcher.py: 删除 _looks_like_close, _contains_close_intent 测试
- test_task_dispatcher.py: 新增 classify_completion() mock 测试
- test_supervisor_main.py: 新增 "reply 包含关闭文本但未执行关闭" 的回归测试
- test_supervisor_main.py: 新增 smart close 端到端测试
```

#### E2: 模块 B 测试 — 线程安全与超时

```
新增:
- test_task_dispatcher.py: 新增超时测试 (mock time, asyncio.timeout)
- test_task_dispatcher.py: 新增并发读写 _tasks 测试
- test_task_dispatcher.py: 新增 _ensure_loaded 延迟加载测试
```

#### E3: 模块 C 测试 — 网关增强

```
新增:
- test_feishu_gateway.py: 新增文件句柄关闭验证测试
- test_supervisor_main.py: 新增 MultiTaskGroup 聚合通知测试
- test_task_dispatcher.py: 新增 on_progress 回调测试
```

#### E4: 模块 D 测试 — 代码重构

```
修改:
- 所有现有测试必须在重构后仍然通过（import 路径可能变化）
- test_scheduler.py: 新增 jitter 验证测试
```

---

## 四、Agent 分配与执行计划

### 角色分配

| Agent | 角色 | 负责模块 | 关键文件 |
|-------|------|----------|----------|
| **Agent 1: PM** | 项目管理 + 代码审查 | 本文档 + 最终审查 | DEVELOPMENT_PLAN.md |
| **Agent 2: Sonnet Intelligence** | 开发 | 模块 A (A1-A4) | router_skill.py, claude_session.py, task_dispatcher.py, main.py |
| **Agent 3: Thread Safety** | 开发 | 模块 B (B1-B3) | task_dispatcher.py |
| **Agent 4: Gateway Monitor** | 开发 | 模块 C (C1-C4) | main.py, feishu_gateway.py, task_dispatcher.py |
| **Agent 5: Code Restructure** | 开发 | 模块 D (D1-D3) | main.py → 拆分, task_dispatcher.py → 拆分 |
| **Agent 6: Tester** | TDD + 测试 | 模块 E (E1-E4) | tests/*.py |
| **Agent 7: Reviewer** | 代码审查 | 所有模块 | 全部文件 |

### 执行顺序 (考虑依赖关系)

```
Phase 1: 并行 (无依赖)
├── Agent 6 (Tester): 先写各模块的失败测试 (TDD Red Phase)
├── Agent 3 (Thread Safety): 模块 B — 独立于其他模块
└── Agent 4 (Gateway): 模块 C2 (文件句柄) + C3 (聚合通知)

Phase 2: 并行 (Phase 1 完成后)
├── Agent 2 (Sonnet Intelligence): 模块 A — 核心修改
│   ├── A1: 重构 prompt (router_skill.py)
│   ├── A2: 结构化输出 (claude_session.py)
│   ├── A3: 删除硬编码 (task_dispatcher.py + main.py)
│   └── A4: 关闭可靠性 (main.py)
└── Agent 4 (Gateway): 模块 C4 (进度推送) — 依赖 B1 的锁

Phase 3: 串行 (Phase 2 完成后)
└── Agent 5 (Code Restructure): 模块 D — 最后做，避免冲突
    ├── D1: 拆分 main.py
    ├── D2: 拆分 task_dispatcher.py
    └── D3: Scheduler jitter

Phase 4: 验证
├── Agent 6 (Tester): 运行全量测试，确认所有通过
├── Agent 7 (Reviewer): 代码审查所有变更
└── Agent 1 (PM): 最终验收
```

### 依赖关系图

```
模块 B (Thread Safety)  ──────┐
                               ├──→ 模块 A (Sonnet Intelligence) ──→ 模块 D (Restructure)
模块 C (Gateway) C2/C3 ───────┘                                        │
                               ├──→ 模块 C4 (Progress Push)     ──────→│
模块 E (Tests) ────────────────┘                                        ↓
                                                              Phase 4: 验证
```

**关键依赖:**
1. 模块 A 依赖模块 B (锁保护) — A 修改的函数需要在 B 的锁保护下工作
2. 模块 C4 依赖模块 B — 进度推送涉及 _tasks 读写
3. 模块 D 必须最后做 — 重构移动文件，会影响 A/B/C 的 import 路径
4. 模块 E 的测试先写 (TDD) — 每个开发模块完成后验证

---

## 五、Git 工作流

### 分支策略

```
master
  └── feat/supervisor-v2 (主开发分支)
       ├── feat/sv2-thread-safety     (模块 B)
       ├── feat/sv2-sonnet-intelligence (模块 A)
       ├── feat/sv2-gateway-monitor    (模块 C)
       └── feat/sv2-restructure        (模块 D)
```

### 合并顺序

1. `feat/sv2-thread-safety` → `feat/supervisor-v2` (先合，其他分支需要)
2. `feat/sv2-gateway-monitor` → `feat/supervisor-v2`
3. `feat/sv2-sonnet-intelligence` → `feat/supervisor-v2`
4. `feat/sv2-restructure` → `feat/supervisor-v2` (最后合)
5. `feat/supervisor-v2` → `master` (全部验证通过后)

### Commit 规范

```
feat(router): restructure routing prompt for better close/followup accuracy
fix(dispatcher): add threading lock to protect global _tasks state
fix(gateway): close file handles in upload_image/upload_file
refactor(main): extract command handlers to commands.py
test(router): add close vs follow_up boundary test cases
```

---

## 六、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| Sonnet 结构化输出增加延迟 | 中 | 中 — 路由延迟增加 | tool_use 模式延迟 <500ms；CLI fallback 保留 |
| 删除硬编码后 Sonnet 分类准确率不如预期 | 中 | 高 — 功能退化 | 保留 /close 命令作为确定性备选路径 |
| 文件拆分导致 import 循环 | 低 | 高 — 启动失败 | 模块 D 最后做，充分测试 |
| 并行开发冲突 (多 Agent 改同一文件) | 高 | 中 — 合并冲突 | Phase 分离 + 锁定文件所有权 |
| Worker 超时误杀长任务 | 中 | 中 — 用户工作丢失 | 默认 30min 足够长；可配置环境变量 |

---

## 七、验收标准

### 功能验收

- [ ] 用户说"关闭/关了/不用了"时，任务状态确实变为 completed
- [ ] 不存在 Sonnet 说"已关闭"但任务实际未关闭的情况
- [ ] 所有硬编码模式匹配代码已删除 (`_CLOSE_PHRASES`, `_INPUT_PHRASES` 等)
- [ ] Worker 超过 30min 自动终止并通知用户
- [ ] 多线程读写 `_tasks` 无竞态条件
- [ ] dispatch_multi 所有子任务完成后有聚合通知
- [ ] 长任务执行中有进度推送（每 30s）
- [ ] 文件句柄泄漏已修复
- [ ] /close 命令仍可正常使用（确定性路径保留）

### 测试验收

- [ ] 所有现有 262 个测试通过
- [ ] 新增测试覆盖所有修改点（预计新增 40+ 测试）
- [ ] 测试覆盖率 ≥ 80%
- [ ] 无 import 错误或循环依赖

### 代码质量

- [ ] 每个文件 ≤ 400 行（重构后）
- [ ] code-reviewer agent 无 CRITICAL/HIGH 问题
- [ ] 无硬编码模式匹配
- [ ] 无全局可变状态无锁读写
