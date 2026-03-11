"""Tests for supervisor.router_skill module."""

from supervisor.router_skill import (
    build_route_prompt,
    build_route_user_prompt,
    ROUTING_EXAMPLES,
    ROUTING_RULES,
    SUPERVISOR_IDENTITY,
)


class TestBuildRoutePrompt:
    """Test the unified route prompt builder."""

    def test_contains_user_message(self):
        prompt = build_route_prompt("帮我分析代码")
        assert "帮我分析代码" in prompt

    def test_contains_identity(self):
        prompt = build_route_prompt("你好")
        assert "Supervisor Hub" in prompt

    def test_contains_routing_rules(self):
        prompt = build_route_prompt("test")
        assert "dispatch" in prompt
        assert "reply" in prompt

    def test_contains_examples(self):
        prompt = build_route_prompt("test")
        assert "reply" in prompt
        assert "dispatch" in prompt

    def test_output_format_specified(self):
        """Prompt must specify the JSON output format."""
        prompt = build_route_prompt("test")
        assert '"action"' in prompt
        assert '"text"' in prompt

    def test_contains_orchestrate(self):
        prompt = build_route_prompt("重构整个项目")
        assert "orchestrate" in prompt


class TestRoutingRules:
    """Test that routing rules cover key scenarios."""

    def test_rules_mention_execution(self):
        assert "执行" in ROUTING_RULES or "execute" in ROUTING_RULES.lower()

    def test_rules_mention_file_access(self):
        assert "文件" in ROUTING_RULES or "file" in ROUTING_RULES.lower()


class TestRoutingExamples:
    """Test that few-shot examples are well-formed."""

    def test_examples_have_reply_cases(self):
        assert "reply" in ROUTING_EXAMPLES

    def test_examples_have_dispatch_cases(self):
        assert "dispatch" in ROUTING_EXAMPLES

    def test_examples_have_orchestrate_cases(self):
        assert "orchestrate" in ROUTING_EXAMPLES


class TestTaskContext:
    """Test that task context is injected into the route prompt."""

    def test_no_tasks_no_context(self):
        prompt = build_route_prompt("test")
        assert "## Currently active tasks" not in prompt
        assert "## Tasks awaiting closure" not in prompt

    def test_active_tasks_shown(self):
        active = [{"id": "4bbe5fbe", "status": "running", "description": "分析代码"}]
        prompt = build_route_prompt("test", active_tasks=active)
        assert "4bbe5fbe" in prompt
        assert "running" in prompt
        assert "分析代码" in prompt

    def test_awaiting_tasks_shown(self):
        awaiting = [{"id": "8b557777", "description": "分析TensorRT-LLM"}]
        prompt = build_route_prompt("test", awaiting_tasks=awaiting)
        assert "8b557777" in prompt
        assert "分析TensorRT-LLM" in prompt

    def test_both_active_and_awaiting(self):
        active = [{"id": "aaaa1111", "status": "running", "description": "任务A"}]
        awaiting = [{"id": "bbbb2222", "description": "任务B"}]
        prompt = build_route_prompt("test", awaiting_tasks=awaiting, active_tasks=active)
        assert "aaaa1111" in prompt
        assert "bbbb2222" in prompt


class TestConversationHistory:
    """Test that conversation history is injected into the route prompt."""

    def test_no_history_no_section(self):
        prompt = build_route_prompt("test")
        assert "## Recent conversation history" not in prompt

    def test_empty_string_no_section(self):
        prompt = build_route_prompt("test", conversation_history="")
        assert "## Recent conversation history" not in prompt

    def test_history_included_when_provided(self):
        history = "User: 你好\nAssistant: 你好！有什么可以帮你的？"
        prompt = build_route_prompt("继续", conversation_history=history)
        assert "## Recent conversation history" in prompt
        assert "User: 你好" in prompt
        assert "Assistant: 你好！有什么可以帮你的？" in prompt

    def test_history_appears_before_user_message(self):
        history = "User: 之前的消息"
        prompt = build_route_prompt("新消息", conversation_history=history)
        history_pos = prompt.index("## Recent conversation history")
        msg_pos = prompt.index("<user_message>")
        assert history_pos < msg_pos

    def test_history_with_tasks(self):
        active = [{"id": "aaaa1111", "status": "running", "description": "任务A"}]
        history = "User: 上一条消息"
        prompt = build_route_prompt(
            "test", active_tasks=active, conversation_history=history,
        )
        assert "## Currently active tasks" in prompt
        assert "## Recent conversation history" in prompt
        assert "aaaa1111" in prompt
        assert "上一条消息" in prompt


class TestCloseAction:
    """Test that close action is defined in routing rules, examples, and format."""

    def test_rules_mention_close_action(self):
        assert 'action = "close"' in ROUTING_RULES

    def test_examples_have_close_cases(self):
        assert '"close"' in ROUTING_EXAMPLES

    def test_close_format_in_prompt(self):
        prompt = build_route_prompt("关闭这个任务")
        assert '"close"' in prompt
        assert "task_id" in prompt

    def test_close_replaces_old_rule_6(self):
        """Old rule 6 told users to use /close <id>. It should be gone."""
        assert '使用 /close' not in ROUTING_RULES
        assert 'tell them to use /close' not in ROUTING_RULES


class TestBatchCloseAction:
    """Test that batch close (close_all) is defined in routing rules and examples."""

    def test_rules_mention_close_all(self):
        assert "close_all" in ROUTING_RULES

    def test_examples_have_close_all_case(self):
        assert '"close_all"' in ROUTING_EXAMPLES

    def test_close_all_format_in_prompt(self):
        prompt = build_route_prompt("把这些任务都关了")
        assert '"close_all"' in prompt

    def test_close_all_few_shot_with_multiple_tasks(self):
        """Few-shot examples should show close_all with multiple awaiting tasks."""
        assert '"close_all"' in ROUTING_EXAMPLES
        assert "全部关" in ROUTING_EXAMPLES or "都关" in ROUTING_EXAMPLES


class TestSupervisorIdentity:
    """Test the identity string for sonnet's reply generation."""

    def test_identity_mentions_supervisor(self):
        assert "Supervisor" in SUPERVISOR_IDENTITY

    def test_identity_mentions_worker(self):
        assert "worker" in SUPERVISOR_IDENTITY


class TestReplyContext:
    """Test that reply_to_task context is injected into the route prompt."""

    def test_no_reply_context_by_default(self):
        prompt = build_route_prompt("test")
        assert "## Reply context" not in prompt

    def test_reply_context_included(self):
        reply_to = {"id": "aabb1122", "description": "分析代码"}
        prompt = build_route_prompt("好的", reply_to_task=reply_to)
        assert "## Reply context" in prompt
        assert "aabb1122" in prompt
        assert "分析代码" in prompt

    def test_reply_context_mentions_close_and_follow_up(self):
        reply_to = {"id": "aabb1122", "description": "分析代码"}
        prompt = build_route_prompt("好的", reply_to_task=reply_to)
        assert "close" in prompt.lower()
        assert "follow_up" in prompt.lower()

    def test_reply_context_appears_before_user_message(self):
        reply_to = {"id": "aabb1122", "description": "分析代码"}
        prompt = build_route_prompt("好的", reply_to_task=reply_to)
        reply_pos = prompt.index("## Reply context")
        msg_pos = prompt.index("<user_message>")
        assert reply_pos < msg_pos

    def test_reply_context_with_history_and_tasks(self):
        awaiting = [{"id": "aabb1122", "description": "分析代码"}]
        reply_to = {"id": "aabb1122", "description": "分析代码"}
        history = "User: 帮我分析代码"
        prompt = build_route_prompt(
            "好的", awaiting_tasks=awaiting,
            conversation_history=history, reply_to_task=reply_to,
        )
        assert "## Reply context" in prompt
        assert "## Tasks awaiting closure" in prompt
        assert "## Recent conversation history" in prompt


class TestEnrichedTaskContext:
    """Test that awaiting tasks include result summary and completion time."""

    def test_result_summary_included(self):
        awaiting = [{"id": "aabb1122", "description": "分析代码",
                      "result_summary": "Found 3 critical issues"}]
        prompt = build_route_user_prompt("test", awaiting_tasks=awaiting)
        assert "Found 3 critical issues" in prompt
        assert "Result summary" in prompt

    def test_completed_at_included(self):
        awaiting = [{"id": "aabb1122", "description": "分析代码",
                      "completed_at": "2m ago"}]
        prompt = build_route_user_prompt("test", awaiting_tasks=awaiting)
        assert "2m ago" in prompt

    def test_no_extra_info_when_not_provided(self):
        awaiting = [{"id": "aabb1122", "description": "分析代码"}]
        prompt = build_route_user_prompt("test", awaiting_tasks=awaiting)
        assert "Result summary" not in prompt
        assert "completed" not in prompt.lower() or "completed" in ROUTING_RULES.lower()


class TestTechnicalCloseDisambiguation:
    """Test that routing rules and examples distinguish task close from technical operations."""

    def test_rules_mention_technical_operations(self):
        assert "数据库连接" in ROUTING_RULES or "technical" in ROUTING_RULES.lower()

    def test_examples_have_technical_dispatch_cases(self):
        """Examples should show technical close operations dispatched, not closed."""
        assert "数据库连接" in ROUTING_EXAMPLES or "nginx" in ROUTING_EXAMPLES

    def test_rules_distinguish_close_from_dispatch(self):
        """Rules must explicitly mention distinguishing task closure from technical ops."""
        assert "dispatch" in ROUTING_RULES
        assert "TECHNICAL" in ROUTING_RULES or "technical" in ROUTING_RULES.lower()


class TestAcknowledgementCloseExamples:
    """Test that examples include short acknowledgements → close patterns."""

    def test_ok_close_example(self):
        assert '"ok"' in ROUTING_EXAMPLES.lower() or '"好的"' in ROUTING_EXAMPLES

    def test_thumbs_up_close_example(self):
        assert "👍" in ROUTING_EXAMPLES

    def test_shoudao_close_example(self):
        assert "收到" in ROUTING_EXAMPLES
