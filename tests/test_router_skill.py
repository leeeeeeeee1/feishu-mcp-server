"""Tests for supervisor.router_skill module."""

from supervisor.router_skill import (
    build_route_prompt,
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

    def test_contains_dispatch_multi(self):
        prompt = build_route_prompt("重构整个项目")
        assert "dispatch_multi" in prompt


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

    def test_examples_have_dispatch_multi_cases(self):
        assert "dispatch_multi" in ROUTING_EXAMPLES


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


class TestSupervisorIdentity:
    """Test the identity string for sonnet's reply generation."""

    def test_identity_mentions_supervisor(self):
        assert "Supervisor" in SUPERVISOR_IDENTITY

    def test_identity_mentions_worker(self):
        assert "worker" in SUPERVISOR_IDENTITY
