"""Offline checks for streamlit_app.py (no API calls).

- static checks (no hardcoded key, absolute DB paths, batch-safe approval routing,
  no nursing-home leftovers)
- business tools against the seeded silver-education DB（学员画像 / 打卡主题 / 课程活动）
- graph regression: batch [list_students, send_notification] must be gated,
  rejection answers every tool_call, approval executes the message
"""
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

os.environ["DEEPSEEK_API_KEY"] = "test-key"

DIR = os.environ.get("KANGYANG_APP_DIR") or str(Path(__file__).resolve().parents[1])
SRC = os.path.join(DIR, "streamlit_app.py")
sys.path.insert(0, DIR)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

# 只执行图/工具/业务库部分，跳过底部的 Streamlit 路由，避免 bare-mode 干扰
source = open(SRC, encoding="utf-8").read()
core = source.split("# ============ 路由 ============")[0]
ns = {"__name__": "yinfa_core", "__file__": SRC}
exec(compile(core, SRC, "exec"), ns)


class BatchFakeLLM:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls

    def invoke(self, messages):
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return AIMessage(content="", tool_calls=self.tool_calls)
        if isinstance(last, ToolMessage):
            return AIMessage(content=f"最终回复：{last.content}")
        raise AssertionError(f"unexpected message: {last}")


BATCH = [
    {"name": "list_students", "args": {}, "id": "call_l", "type": "tool_call"},
    {"name": "send_notification", "args": {"target": "王秀兰", "title": "打卡提醒", "content": "今晚记得完成摄影打卡"}, "id": "call_n", "type": "tool_call"},
]


def open_app(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    app = ns["graph"].compile(checkpointer=SqliteSaver(conn))
    return app, conn


def test_static():
    assert "sk-b7ab" not in source, "hardcoded API key found"
    assert 'os.getenv("DEEPSEEK_API_KEY")' in source
    assert "Path(__file__).resolve().parent" in source, "DB paths must be absolute"
    assert "康养" not in source, "nursing-home leftovers remain"
    assert "yinfa_edu.db" in source and 'sqlite3.connect("yinfa_edu.db")' not in source
    assert 'any(tc["name"] == "send_notification"' in source, "approval gate must check the whole batch"
    assert "route_after_approval" in source and "ToolMessage" in source
    tools = ns.get("TOOL_FEATURES", [])
    assert tools and all(t["func"] is not None for t in tools), "运营工具模块导入失败（TOOL_FEATURES 中存在空 func）"
    assert len({t["key"] for t in tools}) == len(tools), "TOOL_FEATURES 存在重复 key"
    # 共用 LLM 配置防漂移：各智能体模块必须引用 yinfa_zhinao_llm，不得内联 LLM(...)
    llm_modules = ("yinfa_zhinao_topic.py", "yinfa_zhinao_checkin.py", "yinfa_zhinao_renewal.py",
                   "yinfa_zhinao_faq.py", "yinfa_zhinao_event.py", "yinfa_zhinao_curriculum.py",
                   "yinfa_zhinao_grading.py", "yinfa_zhinao_referral.py")
    for name in llm_modules:
        content = open(os.path.join(DIR, name), encoding="utf-8").read()
        assert "build_deepseek_llm" in content and "LLM(" not in content, f"{name} 未使用共用 LLM 配置"
    print("static checks OK")


def test_tools():
    r = ns["query_student"].invoke({"student_name": "王秀兰"})
    assert "高意向" in r and "手机摄影" in r, r
    r = ns["list_students"].invoke({})
    assert "王秀兰" in r and "孙丽华" in r, r
    r = ns["query_checkin_topic"].invoke({})
    assert "窗边" in r, r
    r = ns["list_today_events"].invoke({})
    assert "直播课" in r and "手机摄影" in r, r
    print("silver-education tools OK")


def test_batch_gate(tmp):
    ns["llm_with_tools"] = BatchFakeLLM([dict(tc) for tc in BATCH])
    app, conn = open_app(os.path.join(tmp, "g.db"))
    cfg = {"configurable": {"thread_id": "g"}}
    result = app.invoke({"messages": [("user", "先列出学员，再给王秀兰发消息")]}, cfg)

    snapshot = app.get_state(cfg)
    assert snapshot.tasks and snapshot.tasks[0].interrupts, "batch message bypassed approval"
    assert not any(isinstance(x, ToolMessage) and "消息已发送至" in str(x.content) for x in result["messages"])
    payload = snapshot.tasks[0].interrupts[0].value
    assert [tc["name"] for tc in payload["tool_calls"]] == ["list_students", "send_notification"], payload
    assert payload["tool"] == "send_notification"

    # 拒绝：每个 tool_call 都要有应答，消息不能执行
    result2 = app.invoke(Command(resume=False), cfg)
    batch_ai = next(x for x in result2["messages"] if isinstance(x, AIMessage) and x.tool_calls)
    wanted = {tc["id"] for tc in batch_ai.tool_calls}
    answered = {x.tool_call_id for x in result2["messages"] if isinstance(x, ToolMessage)}
    assert wanted <= answered, f"unanswered tool_calls: {wanted - answered}"
    assert not any("消息已发送至" in str(x.content) for x in result2["messages"])
    conn.close()

    # 批准：消息执行
    ns["llm_with_tools"] = BatchFakeLLM([dict(tc) for tc in BATCH])
    app2, conn2 = open_app(os.path.join(tmp, "g2.db"))
    cfg2 = {"configurable": {"thread_id": "g2"}}
    app2.invoke({"messages": [("user", "发消息")]}, cfg2)
    result3 = app2.invoke(Command(resume=True), cfg2)
    assert any(isinstance(x, ToolMessage) and "消息已发送至" in str(x.content) for x in result3["messages"])
    conn2.close()
    print("batch approval gate OK (interrupt / reject-pairing / approve-execute)")


def test_conversation_records():
    """多会话记录：每条会话独立成记录，标题取首条提问，按最近使用排序，仅本用户可见。"""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = ns["SqliteSaver"](conn)
    app = ns["graph"].compile(checkpointer=saver)

    class PlainLLM:
        def invoke(self, messages):
            return AIMessage(content="好的。")

    ns["llm_with_tools"] = PlainLLM()
    app.invoke({"messages": [("user", "第一段会话的提问")]}, {"configurable": {"thread_id": "alice:c1"}})
    app.invoke({"messages": [("user", "第二段会话的提问")]}, {"configurable": {"thread_id": "alice:c2"}})
    app.invoke({"messages": [("user", "旧版历史会话的提问")]}, {"configurable": {"thread_id": "alice"}})

    records = ns["list_conversation_records"](saver, app, "alice")
    titles = [t for _, t in records]
    assert titles == ["旧版历史会话的提问", "第二段会话的提问", "第一段会话的提问"], titles
    assert [r for r, _ in records][-1].startswith("alice:"), records
    assert ns["set_conversation_title"](saver, "alice:c1", "自定义标题一")
    renamed = dict(ns["list_conversation_records"](saver, app, "alice"))
    assert renamed["alice:c1"] == "自定义标题一", renamed
    assert ns["list_conversation_records"](saver, app, "bob") == []
    conn.close()
    print("conversation records OK (多会话 + 旧版线程兼容，标题取首条提问，按最近排序)")


def main():
    test_static()
    test_tools()
    tmp = tempfile.mkdtemp(prefix="yinfa-")
    try:
        test_batch_gate(tmp)
        test_conversation_records()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("YINFA OFFLINE CHECKS PASSED")


if __name__ == "__main__":
    main()
