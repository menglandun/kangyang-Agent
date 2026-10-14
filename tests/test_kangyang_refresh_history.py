"""M2-4 residual: 会话记录切回后从 checkpoint 重建真实内容（多会话新语义）。

Instance #1 logs in as bob and asks a question (real API). Instance #2 is a brand-new
AppTest session: 新登录默认开一条空会话，从侧栏「会话记录」切回后必须重建出问题与回答。
"""
import os
import sqlite3
import time
from pathlib import Path

if not os.environ.get("DEEPSEEK_API_KEY"):
    raise SystemExit("run with DEEPSEEK_API_KEY set")

DIR = os.environ.get("KANGYANG_APP_DIR") or str(Path(__file__).resolve().parents[1])
APP = f"{DIR}/streamlit_app.py"
DB = f"{DIR}/checkpoints.db"

from streamlit.testing.v1 import AppTest

QUESTION = "查一下王秀兰的学员画像"
EXPECT = "高意向"


def fail(msg):
    print("FAIL:", msg)
    raise SystemExit(1)


def login(at, username, password):
    at.run()
    if at.exception:
        fail(f"initial run: {[getattr(e, 'value', e) for e in at.exception]}")
    at.text_input[0].set_value(username)
    at.text_input[1].set_value(password)
    at.button[0].click().run()
    if at.exception:
        fail(f"login run: {[getattr(e, 'value', e) for e in at.exception]}")


def markdown_text(at):
    return "\n".join(m.value for m in at.markdown)


def bob_rows():
    if not os.path.exists(DB):
        return 0
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    n = 0
    for table in ("checkpoints", "writes"):
        n += conn.execute(f"select count(*) from {table} where thread_id LIKE 'bob:%'").fetchone()[0]
    conn.close()
    return n


pre = bob_rows()

# 会话 1：bob 登录并真实提问（写入 bob 线程 checkpoint）
at1 = AppTest.from_file(APP, default_timeout=180)
login(at1, "bob", "bob123")
at1_text = markdown_text(at1) + "\n" + "\n".join(m.value for m in at1.sidebar.markdown)
if "🤖 银发智脑・兴趣教育智慧中台" not in at1_text:
    fail("bob login failed: main header missing")

at1.chat_input[0].set_value(QUESTION).run()
if at1.exception:
    fail(f"chat run: {[getattr(e, 'value', e) for e in at1.exception]}")
if EXPECT not in markdown_text(at1):
    fail("instance 1 answer missing expected content")
print("instance 1: bob asked a question (real API) OK")

# 会话 2：全新的 AppTest 实例 = 新浏览器会话（session_state 为空）
at2 = AppTest.from_file(APP, default_timeout=180)
login(at2, "bob", "bob123")
at2_text = markdown_text(at2) + "\n" + "\n".join(m.value for m in at2.sidebar.markdown)
if "🤖 银发智脑・兴趣教育智慧中台" not in at2_text:
    fail("instance 2 login failed")
if QUESTION in markdown_text(at2):
    fail("fresh login should start an empty conversation (no auto-resume)")

record_btn = None
for b in at2.button:
    if QUESTION in b.label:
        record_btn = b
        break
if record_btn is None:
    fail(f"conversation record entry missing: {[b.label for b in at2.button]}")
record_btn.click().run()
if at2.exception:
    fail(f"switch record run: {[getattr(e, 'value', e) for e in at2.exception]}")

rebuilt = markdown_text(at2)
if QUESTION not in rebuilt:
    fail(f"rebuilt history missing the user question; got: {rebuilt[:400]}")
if EXPECT not in rebuilt:
    fail(f"rebuilt history missing the assistant answer; got: {rebuilt[:400]}")
print("M2-4 content-level rebuild OK: 会话记录切回后重建问题与回答")

# 若 bob 线程是本次测试首次创建，清理掉，避免污染演示数据
if pre == 0:
    conn = sqlite3.connect(DB)
    conn.execute("DELETE FROM checkpoints WHERE thread_id LIKE 'bob:%'")
    conn.execute("DELETE FROM writes WHERE thread_id LIKE 'bob:%'")
    conn.commit()
    conn.close()
    print("cleanup OK (bob thread created by this test)")
else:
    print("note: bob thread already existed; test messages were appended and left in place")

print("KANGYANG REFRESH-REBUILD PASSED")
