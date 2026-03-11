"""Tests for supervisor.task_dispatcher module."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock

from supervisor.task_dispatcher import (
    Task,
    _looks_like_needs_input,
    _looks_like_close,
    _contains_close_intent,
    _build_cmd,
    _reset,
    dispatch,
    resume_task,
    submit_review,
    skip_review,
    get_task,
    list_tasks,
    list_daemons,
    stop_daemon,
    cancel_task,
    close_task,
    close_tasks,
    follow_up_async,
    get_awaiting_closure,
    get_tasks_text,
    get_daemons_text,
    get_review_pending,
    _set_status,
    _format_task,
)


class _AsyncLineIterator:
    """Async iterator that yields lines from a bytes buffer."""
    def __init__(self, data: bytes):
        self._lines = data.split(b"\n")
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        return line


def _make_proc_mock(stdout_data=b"", stderr_data=b"", returncode=0):
    """Create a mock asyncio subprocess that supports both streaming and communicate."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout_data, stderr_data))
    proc.stdout = _AsyncLineIterator(stdout_data)
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=stderr_data)
    proc.wait = AsyncMock(return_value=returncode)
    proc.returncode = returncode
    proc.kill = MagicMock()
    return proc


def _json_result(result_text="OK", session_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890"):
    """Build a stream-json result line (used by streaming _run_claude)."""
    return json.dumps({
        "type": "result",
        "subtype": "success",
        "result": result_text,
        "session_id": session_id,
    }).encode()


# ── Heuristic tests ──


class TestInputHeuristic:
    def test_no_question_mark(self):
        assert _looks_like_needs_input("I finished the task.") is False

    def test_question_but_no_phrase(self):
        assert _looks_like_needs_input("What is 2+2?") is False

    def test_looks_like_input_please(self):
        assert _looks_like_needs_input("Could you please clarify?") is True

    def test_looks_like_input_should_i(self):
        assert _looks_like_needs_input("Should I proceed with this?") is True

    def test_looks_like_input_which(self):
        assert _looks_like_needs_input("Which file do you mean?") is True

    def test_looks_like_input_confirm(self):
        assert _looks_like_needs_input("Please confirm this action?") is True


# ── Close intent heuristic tests ──


class TestCloseHeuristic:
    def test_chinese_acknowledgements(self):
        assert _looks_like_close("好的") is True
        assert _looks_like_close("收到") is True
        assert _looks_like_close("可以了") is True
        assert _looks_like_close("没问题") is True
        assert _looks_like_close("不用了") is True
        assert _looks_like_close("就这样") is True
        assert _looks_like_close("完成") is True

    def test_english_acknowledgements(self):
        assert _looks_like_close("ok") is True
        assert _looks_like_close("OK") is True
        assert _looks_like_close("thanks") is True
        assert _looks_like_close("done") is True
        assert _looks_like_close("lgtm") is True

    def test_with_trailing_punctuation(self):
        assert _looks_like_close("好的。") is True
        assert _looks_like_close("收到！") is True
        assert _looks_like_close("ok!") is True
        assert _looks_like_close("谢谢~") is True

    def test_with_emoji(self):
        assert _looks_like_close("👍") is True

    def test_long_message_not_close(self):
        assert _looks_like_close("好的，但是我还有个问题想问一下关于这个接口的实现") is False

    def test_question_not_close(self):
        assert _looks_like_close("好的，还有别的问题吗？") is False

    def test_actual_follow_up_not_close(self):
        assert _looks_like_close("帮我再加个鉴权") is False
        assert _looks_like_close("这个接口还需要改一下返回格式") is False

    def test_empty_string(self):
        assert _looks_like_close("") is False

    def test_whitespace_handling(self):
        assert _looks_like_close("  好的  ") is True
        assert _looks_like_close("  ok  ") is True

    def test_negation_not_close(self):
        assert _looks_like_close("not ok") is False

    def test_status_report_not_close(self):
        assert _looks_like_close("I am done") is False
        assert _looks_like_close("all done") is False

    def test_embedded_phrase_not_close(self):
        assert _looks_like_close("ok but") is False
        assert _looks_like_close("lgtm bro") is False


class TestContainsCloseIntent:
    """Tests for _contains_close_intent — detects close keywords in longer text."""

    def test_xxx_closed(self):
        assert _contains_close_intent("检查系统日志那个关闭了") is True

    def test_xxx_guan_le(self):
        assert _contains_close_intent("分析代码那个关了") is True

    def test_ba_xxx_guan_diao(self):
        assert _contains_close_intent("把那个任务关掉") is True

    def test_jieshu(self):
        assert _contains_close_intent("分析代码那个结束吧") is True

    def test_buyongle(self):
        assert _contains_close_intent("那个不用了") is True

    def test_wanshile(self):
        assert _contains_close_intent("完事了") is True

    def test_keyi_guan_le(self):
        assert _contains_close_intent("可以关了") is True

    def test_english_close(self):
        assert _contains_close_intent("please close that task") is True

    def test_english_done_with_it(self):
        assert _contains_close_intent("I'm done with it") is True

    def test_guanbi_ba(self):
        assert _contains_close_intent("那个任务关闭吧") is True

    def test_guanbi_zhege(self):
        assert _contains_close_intent("关闭这个") is True

    def test_jieshu_le(self):
        assert _contains_close_intent("任务结束了") is True

    def test_jieshu_renwu(self):
        assert _contains_close_intent("结束任务") is True

    def test_false_positive_close_connection(self):
        assert _contains_close_intent("关闭连接后重试") is False

    def test_false_positive_close_port(self):
        assert _contains_close_intent("关闭端口") is False

    def test_false_positive_close_service(self):
        assert _contains_close_intent("关闭服务") is False

    def test_false_positive_close_process(self):
        assert _contains_close_intent("关掉进程") is False

    def test_false_positive_process_guan_diao(self):
        """Noun-before-verb: '进程关掉' should NOT be close intent."""
        assert _contains_close_intent("帮我把nginx进程关掉") is False

    def test_false_positive_bare_jieshu(self):
        """Bare '结束' without task suffix should NOT match."""
        assert _contains_close_intent("死循环不会结束") is False

    def test_false_positive_jieshu_process(self):
        assert _contains_close_intent("结束进程") is False

    def test_no_close_intent(self):
        assert _contains_close_intent("帮我再看一下这个结果") is False

    def test_empty_string(self):
        assert _contains_close_intent("") is False


# ── Build command tests ──


class TestBuildCmd:
    def test_basic_command(self):
        cmd = _build_cmd("hello world")
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "hello world" in cmd
        assert "--model" in cmd
        assert "opus" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd

    def test_with_session_id(self):
        cmd = _build_cmd("test", session_id="abc-123")
        assert "--resume" in cmd
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == "abc-123"

    def test_without_session_id(self):
        cmd = _build_cmd("test")
        assert "--resume" not in cmd


# ── Task creation and dispatch ──


class TestDispatch:
    def setup_method(self):
        _reset()

    def test_task_creation_and_id(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result())):
                task = await dispatch("do something", cwd="/tmp")
                assert len(task.id) == 36  # UUID format
                assert task.prompt == "do something"
                assert task.task_type == "oneshot"
                assert task.cwd == "/tmp"
                assert task.created_at > 0
                # Let the background worker finish
                await asyncio.sleep(0.1)

        asyncio.run(_test())

    def test_status_pending_to_running_to_done(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("All done."))):
                task = await dispatch("work")
                # Let the background worker finish
                await asyncio.sleep(0.1)
                assert task.status == "awaiting_closure"
                assert task.result == "All done."
                assert task.started_at > 0
                assert task.finished_at > 0

        asyncio.run(_test())

    def test_full_review_flow(self):
        """pending -> running -> done -> review -> learning -> completed"""
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("Result text"))):
                task = await dispatch("work")
                await asyncio.sleep(0.1)
                assert task.status == "awaiting_closure"

                # Manually advance to review (simulating what caller does)
                _set_status(task, "review")
                assert task.status == "review"

                feedback = submit_review(task.id, "Great job!")
                assert feedback == "Great job!"
                assert task.status == "completed"

        asyncio.run(_test())

    def test_skip_review_flow(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("Done"))):
                task = await dispatch("work")
                await asyncio.sleep(0.1)
                assert task.status == "awaiting_closure"

                # close_task handles awaiting_closure -> completed
                msg = close_task(task.id)
                assert "closed" in msg
                assert task.status == "completed"

        asyncio.run(_test())

    def test_failure_sets_failed(self):
        async def _test():
            proc = _make_proc_mock(stderr_data=b"boom", returncode=1)
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=proc):
                task = await dispatch("bad task")
                await asyncio.sleep(0.1)
                assert task.status == "failed"
                assert task.error == "boom"

        asyncio.run(_test())

    def test_waiting_for_input_heuristic(self):
        async def _test():
            result = _json_result("Which option should I pick?")
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(result)):
                task = await dispatch("ambiguous task")
                await asyncio.sleep(0.1)
                assert task.status == "waiting_for_input"

        asyncio.run(_test())


# ── Daemon tasks ──


class TestDaemons:
    def setup_method(self):
        _reset()

    def test_daemon_listing(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result())):
                await dispatch("oneshot task", task_type="oneshot")
                await dispatch("daemon task", task_type="daemon")
                await asyncio.sleep(0.1)

                all_tasks = list_tasks()
                assert len(all_tasks) == 2

                daemons = list_daemons()
                assert len(daemons) == 1
                assert daemons[0].task_type == "daemon"

        asyncio.run(_test())

    def test_stop_daemon(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result())):
                task = await dispatch("watch things", task_type="daemon")
                await asyncio.sleep(0.1)

                msg = stop_daemon(task.id)
                assert "stopped" in msg
                assert task.status == "cancelled"

        asyncio.run(_test())

    def test_daemon_auto_restart(self):
        """A daemon that fails should auto-restart up to max_retries.

        Each _run_claude attempt triggers streaming then non-streaming fallback,
        so 2 subprocess calls per _run_claude failure.
        """
        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # 2 failures × 2 modes (streaming + non-streaming) = 4 calls,
            # then 5th call (streaming retry) succeeds
            if call_count < 5:
                return _make_proc_mock(stderr_data=b"error", returncode=1)
            return _make_proc_mock(_json_result("finally worked"))

        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       side_effect=mock_exec):
                task = await dispatch("flaky daemon", task_type="daemon")
                await asyncio.sleep(0.3)
                # After 2 full failures and 1 success it should be awaiting_closure
                assert task.status == "awaiting_closure"
                assert task.retries == 2

        asyncio.run(_test())


# ── Cancel, get, resume ──


class TestQueryAndControl:
    def setup_method(self):
        _reset()

    def test_cancel_task(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result())):
                task = await dispatch("work")
                msg = cancel_task(task.id)
                assert "cancelled" in msg
                assert task.status == "cancelled"

        asyncio.run(_test())

    def test_get_task(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result())):
                task = await dispatch("work")
                found = get_task(task.id)
                assert found.id == task.id

        asyncio.run(_test())

    def test_get_task_unknown(self):
        try:
            get_task("nonexistent")
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_resume_task(self):
        async def _test():
            # First, create a task that ends up waiting_for_input
            result_needing_input = _json_result("Which file should I use please?")
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(result_needing_input)):
                task = await dispatch("ambiguous")
                await asyncio.sleep(0.1)
                assert task.status == "waiting_for_input"

            # Now resume it
            resume_result = _json_result("Done, used file A.")
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(resume_result)):
                text = await resume_task(task.id, "Use file A")
                assert text == "Done, used file A."
                assert task.status == "done"

        asyncio.run(_test())

    def test_resume_wrong_status(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result())):
                task = await dispatch("work")
                await asyncio.sleep(0.1)
                try:
                    await resume_task(task.id, "hello")
                    assert False, "Should have raised"
                except ValueError:
                    pass

        asyncio.run(_test())

    def test_get_review_pending(self):
        async def _test():
            result_input = _json_result("Should I confirm please?")
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(result_input)):
                t1 = await dispatch("task1")
                await asyncio.sleep(0.1)

            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("All done"))):
                t2 = await dispatch("task2")
                await asyncio.sleep(0.1)
                _set_status(t2, "review")

            pending = get_review_pending()
            ids = {t.id for t in pending}
            assert t1.id in ids  # waiting_for_input
            assert t2.id in ids  # review

        asyncio.run(_test())


# ── Semaphore limiting ──


class TestSemaphore:
    def setup_method(self):
        _reset()

    def test_worker_semaphore_limits_concurrency(self):
        """Dispatch more tasks than max workers; only N should run concurrently."""
        running_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def slow_exec(*args, **kwargs):
            nonlocal running_count, max_concurrent
            async with lock:
                running_count += 1
                if running_count > max_concurrent:
                    max_concurrent = running_count
            await asyncio.sleep(0.1)
            async with lock:
                running_count -= 1
            return _make_proc_mock(_json_result())

        async def _test():
            with patch("supervisor.task_dispatcher.SUPERVISOR_MAX_WORKERS", 2):
                # Reset semaphore so it picks up the patched value
                import supervisor.task_dispatcher as td
                td._worker_semaphore = asyncio.Semaphore(2)

                with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                           side_effect=slow_exec):
                    tasks = []
                    for i in range(5):
                        t = await dispatch(f"task {i}")
                        tasks.append(t)

                    # Wait for all to finish
                    await asyncio.sleep(1.0)

                    assert max_concurrent <= 2

        asyncio.run(_test())


# ── Text formatting ──


class TestFormatting:
    def setup_method(self):
        _reset()

    def test_get_tasks_text_empty(self):
        assert get_tasks_text() == "No tasks."

    def test_get_daemons_text_empty(self):
        assert get_daemons_text() == "No daemon tasks."

    def test_get_tasks_text_with_tasks(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("result"))):
                await dispatch("build feature")
                await asyncio.sleep(0.1)
                text = get_tasks_text()
                assert "build feature" in text
                assert "AWAIT_CLOSE" in text

        asyncio.run(_test())

    def test_format_task_truncation(self):
        task = Task(
            id="12345678-1234-1234-1234-123456789abc",
            prompt="a very long prompt that should be truncated in the output line",
            task_type="oneshot",
            status="done",
            result="x" * 200,
            started_at=time.time() - 10,
            finished_at=time.time(),
        )
        line = _format_task(task)
        assert "12345678" in line
        assert "..." in line  # result truncated
        assert "10s" in line or "11s" in line  # elapsed time


# ── Error handling ──


class TestErrorHandling:
    def setup_method(self):
        _reset()

    def test_submit_review_wrong_status(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result())):
                task = await dispatch("work")
                await asyncio.sleep(0.1)
                try:
                    submit_review(task.id, "feedback")
                    assert False, "Should have raised"
                except ValueError as e:
                    assert "not in review" in str(e)

        asyncio.run(_test())

    def test_stop_non_daemon(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result())):
                task = await dispatch("work", task_type="oneshot")
                try:
                    stop_daemon(task.id)
                    assert False, "Should have raised"
                except ValueError as e:
                    assert "not a daemon" in str(e)

        asyncio.run(_test())

    def test_cancel_unknown_task(self):
        try:
            cancel_task("nonexistent-id")
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_claude_not_found(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       side_effect=FileNotFoundError):
                task = await dispatch("work")
                await asyncio.sleep(0.1)
                assert task.status == "failed"
                assert "not found" in task.error

        asyncio.run(_test())


# ── Close task ──


class TestCloseTask:
    def setup_method(self):
        _reset()

    def test_close_awaiting_closure(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("Result"))):
                task = await dispatch("work")
                await asyncio.sleep(0.1)
                assert task.status == "awaiting_closure"

                msg = close_task(task.id)
                assert "closed" in msg
                assert task.status == "completed"

        asyncio.run(_test())

    def test_close_wrong_status(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("Which file?"))):
                task = await dispatch("work")
                await asyncio.sleep(0.1)
                assert task.status == "waiting_for_input"

                try:
                    close_task(task.id)
                    assert False, "Should have raised"
                except ValueError as e:
                    assert "cannot be closed" in str(e)

        asyncio.run(_test())

    def test_close_unknown_task(self):
        try:
            close_task("nonexistent-id")
            assert False, "Should have raised"
        except ValueError:
            pass


# ── Batch close tasks ──


class TestCloseTasks:
    def setup_method(self):
        _reset()

    def test_close_multiple_tasks(self):
        """close_tasks() closes multiple awaiting_closure tasks at once."""
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("Done"))):
                t1 = await dispatch("task1")
                t2 = await dispatch("task2")
                t3 = await dispatch("task3")
                await asyncio.sleep(0.1)
                assert t1.status == "awaiting_closure"
                assert t2.status == "awaiting_closure"
                assert t3.status == "awaiting_closure"

                results = close_tasks([t1.id, t2.id, t3.id])
                assert len(results) == 3
                assert all("closed" in r for r in results)
                assert t1.status == "completed"
                assert t2.status == "completed"
                assert t3.status == "completed"

        asyncio.run(_test())

    def test_close_tasks_partial_failure(self):
        """close_tasks() reports per-task errors without stopping batch."""
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("Done"))):
                t1 = await dispatch("task1")
                await asyncio.sleep(0.1)
                assert t1.status == "awaiting_closure"

                results = close_tasks([t1.id, "nonexistent-id"])
                assert len(results) == 2
                assert "closed" in results[0]
                assert "Error" in results[1] or "Unknown" in results[1]
                assert t1.status == "completed"

        asyncio.run(_test())

    def test_close_tasks_empty_list(self):
        """close_tasks() with empty list returns empty results."""
        results = close_tasks([])
        assert results == []

    def test_close_tasks_wrong_status_included(self):
        """close_tasks() skips tasks that cannot be closed, reports error."""
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("Which file?"))):
                t1 = await dispatch("task1")
                await asyncio.sleep(0.1)
                assert t1.status == "waiting_for_input"

                results = close_tasks([t1.id])
                assert len(results) == 1
                assert "cannot be closed" in results[0]

        asyncio.run(_test())

    def test_close_tasks_mixed_statuses(self):
        """close_tasks() handles mix of closable and non-closable tasks."""
        async def _test():
            proc_ok = _make_proc_mock(_json_result("Done"))
            proc_input = _make_proc_mock(_json_result("Which file?"))
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       side_effect=[proc_ok, proc_input]):
                t1 = await dispatch("task1")
                t2 = await dispatch("task2")
                await asyncio.sleep(0.1)
                assert t1.status == "awaiting_closure"
                assert t2.status == "waiting_for_input"

                results = close_tasks([t1.id, t2.id])
                assert len(results) == 2
                assert "closed" in results[0]
                assert "cannot be closed" in results[1]
                assert t1.status == "completed"
                assert t2.status == "waiting_for_input"

        asyncio.run(_test())


# ── Get awaiting closure ──


class TestGetAwaitingClosure:
    def setup_method(self):
        _reset()

    def test_returns_awaiting_tasks(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("Done"))):
                t1 = await dispatch("task1")
                t2 = await dispatch("task2")
                await asyncio.sleep(0.1)

                awaiting = get_awaiting_closure()
                ids = {t.id for t in awaiting}
                assert t1.id in ids
                assert t2.id in ids

        asyncio.run(_test())

    def test_excludes_closed_tasks(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("Done"))):
                t1 = await dispatch("task1")
                t2 = await dispatch("task2")
                await asyncio.sleep(0.1)

                close_task(t1.id)
                awaiting = get_awaiting_closure()
                ids = {t.id for t in awaiting}
                assert t1.id not in ids
                assert t2.id in ids

        asyncio.run(_test())

    def test_empty_when_no_tasks(self):
        assert get_awaiting_closure() == []


# ── Follow-up async ──


class TestFollowUpAsync:
    def setup_method(self):
        _reset()

    def test_follow_up_on_awaiting_task(self):
        async def _test():
            # Create a task that finishes normally
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("Initial result", "550e8400-e29b-41d4-a716-446655440000"))):
                task = await dispatch("work")
                await asyncio.sleep(0.1)
                assert task.status == "awaiting_closure"
                assert task.session_id == "550e8400-e29b-41d4-a716-446655440000"

            # Follow up on it
            follow_result = _json_result("Follow-up answer", "550e8400-e29b-41d4-a716-446655440000")
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(follow_result)):
                result = await follow_up_async(task.id, "Tell me more")
                assert result == "Follow-up answer"
                assert task.status == "awaiting_closure"

        asyncio.run(_test())

    def test_follow_up_wrong_status(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("Which option?"))):
                task = await dispatch("work")
                await asyncio.sleep(0.1)
                assert task.status == "waiting_for_input"

                try:
                    await follow_up_async(task.id, "hello")
                    assert False, "Should have raised"
                except ValueError as e:
                    assert "not awaiting closure" in str(e)

        asyncio.run(_test())


class TestLoadTasksPersistence:
    """Test _load_tasks handles edge cases: None floats, orphaned .tmp files."""

    def setup_method(self):
        _reset()

    def test_load_tasks_null_float_fields(self, tmp_path):
        """_load_tasks should not crash when float fields are None in JSON."""
        from supervisor.task_dispatcher import _load_tasks, _tasks, _TASKS_FILE
        import supervisor.task_dispatcher as td

        tasks_file = tmp_path / "tasks.json"
        data = {
            "task-123": {
                "id": "task-123",
                "prompt": "test",
                "task_type": "oneshot",
                "status": "awaiting_closure",
                "created_at": 1000.0,
                "started_at": None,
                "finished_at": None,
            }
        }
        tasks_file.write_text(json.dumps(data))

        # Temporarily swap the file path
        original = td._TASKS_FILE
        td._TASKS_FILE = tasks_file
        _tasks.clear()
        try:
            _load_tasks()
            assert "task-123" in _tasks
            t = _tasks["task-123"]
            assert t.started_at == 0.0
            assert t.finished_at == 0.0
        finally:
            td._TASKS_FILE = original
            _tasks.clear()

    def test_load_tasks_orphaned_tmp_recovery(self, tmp_path):
        """_load_tasks should recover from orphaned .tmp file."""
        from supervisor.task_dispatcher import _load_tasks, _tasks
        import supervisor.task_dispatcher as td

        tasks_file = tmp_path / "tasks.json"
        tmp_file = tmp_path / "tasks.tmp"
        data = {
            "task-456": {
                "id": "task-456",
                "prompt": "test",
                "task_type": "oneshot",
                "status": "completed",
                "created_at": 1000.0,
                "started_at": 1001.0,
                "finished_at": time.time(),
            }
        }
        tmp_file.write_text(json.dumps(data))

        original = td._TASKS_FILE
        td._TASKS_FILE = tasks_file
        _tasks.clear()
        try:
            _load_tasks()
            # tmp should have been renamed to tasks_file
            assert tasks_file.exists()
            assert "task-456" in _tasks
        finally:
            td._TASKS_FILE = original
            _tasks.clear()

    def test_load_tasks_running_marked_failed(self, tmp_path):
        """Tasks that were running at crash time should be marked failed."""
        from supervisor.task_dispatcher import _load_tasks, _tasks
        import supervisor.task_dispatcher as td

        tasks_file = tmp_path / "tasks.json"
        data = {
            "task-789": {
                "id": "task-789",
                "prompt": "test",
                "task_type": "oneshot",
                "status": "running",
                "created_at": 1000.0,
                "started_at": 1001.0,
                "finished_at": 0.0,
            }
        }
        tasks_file.write_text(json.dumps(data))

        original = td._TASKS_FILE
        td._TASKS_FILE = tasks_file
        _tasks.clear()
        try:
            _load_tasks()
            assert _tasks["task-789"].status == "failed"
            assert "restarted" in _tasks["task-789"].error.lower()
        finally:
            td._TASKS_FILE = original
            _tasks.clear()


class TestDispatchWithDescription:
    """Test that dispatch() accepts and uses an explicit description."""

    def setup_method(self):
        _reset()

    def test_explicit_description_used(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("OK"))):
                task = await dispatch(
                    "You are a worker agent...\nTask: Analyze code",
                    description="Analyze code",
                )
                assert task.description == "Analyze code"
                await asyncio.sleep(0.1)
        asyncio.run(_test())

    def test_empty_description_falls_back_to_auto(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("OK"))):
                task = await dispatch("Simple request here")
                assert task.description == "Simple request here"
                await asyncio.sleep(0.1)
        asyncio.run(_test())

    def test_explicit_description_avoids_worker_preamble(self):
        async def _test():
            with patch("supervisor.task_dispatcher.asyncio.create_subprocess_exec",
                       return_value=_make_proc_mock(_json_result("OK"))):
                task = await dispatch(
                    "You are a worker agent in a container.\nTask: Build X",
                    description="Build X",
                )
                assert task.description == "Build X"
                assert "worker agent" not in task.description
                await asyncio.sleep(0.1)
        asyncio.run(_test())
