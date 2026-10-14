"""Residual-risk tests (offline): multi-tab resume safety + multi-process SQLite contention.

- concurrent resume via the app's lock+recheck path: exactly one session resumes
- raw concurrent double-resume: message still executes only once, no deadlock
- two processes writing the same SQLite file: second waits (busy timeout) instead of
  failing with "database is locked"; no lost writes
- static checks: threading lock + connect timeout are present in the app
"""
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ["DEEPSEEK_API_KEY"] = "test-key"

DIR = os.environ.get("KANGYANG_APP_DIR") or str(Path(__file__).resolve().parents[1])
SRC = os.path.join(DIR, "streamlit_app.py")
sys.path.insert(0, DIR)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

source = open(SRC, encoding="utf-8").read()
core = source.split("# ============ 路由 ============")[0]
ns = {"__name__": "yinfa_core", "__file__": SRC}
exec(compile(core, SRC, "exec"), ns)

NOTICE_TC = {"name": "send_notification", "args": {"target": "王秀兰", "title": "并发测试", "content": "x"}, "id": "call_n", "type": "tool_call"}


class FakeLLM:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls

    def invoke(self, messages):
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return AIMessage(content="", tool_calls=self.tool_calls)
        if isinstance(last, ToolMessage):
            return AIMessage(content=f"最终回复：{last.content}")
        raise AssertionError(f"unexpected message: {last}")


def open_app(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
    app = ns["graph"].compile(checkpointer=SqliteSaver(conn))
    return app, conn


def make_pending(app, cfg):
    ns["llm_with_tools"] = FakeLLM([dict(NOTICE_TC)])
    app.invoke({"messages": [("user", "发消息")]}, cfg)
    assert app.get_state(cfg).tasks[0].interrupts


def count_sent(app, cfg):
    msgs = app.get_state(cfg).values["messages"]
    return sum(1 for m in msgs if isinstance(m, ToolMessage) and "消息已发送至" in str(m.content))


def test_concurrent_resume_with_app_lock(tmp):
    app, conn = open_app(os.path.join(tmp, "lock.db"))
    cfg = {"configurable": {"thread_id": "lock"}}
    make_pending(app, cfg)

    out = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        with ns["resume_lock"]("lock"):
            fresh = app.get_state(cfg)
            if not (fresh.tasks and fresh.tasks[0].interrupts):
                out.append("skipped")
                return
            app.invoke(Command(resume=True), cfg)
            out.append("resumed")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert all(not t.is_alive() for t in threads), "deadlock in concurrent resume"
    assert sorted(out) == ["resumed", "skipped"], out
    assert count_sent(app, cfg) == 1
    conn.close()
    print("concurrent resume OK (lock + recheck: 1 resumed / 1 skipped, single execution)")


def test_raw_double_resume_single_execution(tmp):
    app, conn = open_app(os.path.join(tmp, "raw.db"))
    cfg = {"configurable": {"thread_id": "raw"}}
    make_pending(app, cfg)

    errors = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        try:
            app.invoke(Command(resume=True), cfg)
        except Exception as exc:  # 版本相关：重复 resume 可能报错，但绝不能重复执行
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert all(not t.is_alive() for t in threads), "deadlock in raw double resume"
    assert count_sent(app, cfg) == 1, "message executed more than once"
    conn.close()
    print(f"raw double resume OK (single execution; extra errors: {errors})")


def test_multiprocess_sqlite_contention(tmp):
    db = os.path.join(tmp, "contend.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t(v INT)")
    conn.commit()
    conn.close()

    writer1 = (
        "import sqlite3,sys,time\n"
        "c=sqlite3.connect(sys.argv[1],timeout=10)\n"
        "c.execute('BEGIN IMMEDIATE')\n"
        "time.sleep(1.2)\n"
        "c.execute('INSERT INTO t VALUES(1)')\n"
        "c.commit()\n"
    )
    writer2 = (
        "import sqlite3,sys,time\n"
        "c=sqlite3.connect(sys.argv[1],timeout=10)\n"
        "t0=time.time()\n"
        "c.execute('INSERT INTO t VALUES(2)')\n"
        "c.commit()\n"
        "print(time.time()-t0)\n"
    )
    p1 = subprocess.Popen([sys.executable, "-c", writer1, db])
    time.sleep(0.25)
    p2 = subprocess.run([sys.executable, "-c", writer2, db], capture_output=True, text=True, timeout=20)
    p1.wait(timeout=20)
    assert p1.returncode == 0, f"writer1 failed: {p1.returncode}"
    assert p2.returncode == 0, f"writer2 failed: {p2.stderr}"
    waited = float(p2.stdout.strip())
    assert waited > 0.5, f"writer2 did not wait for the lock ({waited:.2f}s)"

    conn = sqlite3.connect(db)
    rows = sorted(r[0] for r in conn.execute("select v from t"))
    conn.close()
    assert rows == [1, 2], f"lost writes: {rows}"
    print(f"multi-process sqlite contention OK (writer2 waited {waited:.2f}s, no lost writes)")


def test_static_hardening():
    assert "import threading" in source and "def resume_lock(" in source
    assert "timeout=10" in source, "checkpointer connect must set busy timeout"
    assert "with resume_lock(user_id)" in source, "approval handlers must hold the per-thread lock"
    print("static hardening checks OK")


def main():
    tmp = tempfile.mkdtemp(prefix="yinfa-residual-")
    try:
        test_static_hardening()
        test_concurrent_resume_with_app_lock(tmp)
        test_raw_double_resume_single_execution(tmp)
        test_multiprocess_sqlite_contention(tmp)
        print("KANGYANG RESIDUAL CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
