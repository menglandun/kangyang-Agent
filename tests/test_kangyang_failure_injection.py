"""M2-5 residual: real API failure injection (invalid key -> 401) must be handled gracefully.

The app must show a Chinese error, must not render an exception page, and must not
leave the failed user message stuck in the session history.
"""
import os
import sqlite3
import time
from pathlib import Path

os.environ["DEEPSEEK_API_KEY"] = "sk-invalid-key-for-failure-test"

DIR = os.environ.get("KANGYANG_APP_DIR") or str(Path(__file__).resolve().parents[1])
APP = f"{DIR}/streamlit_app.py"
DB = f"{DIR}/checkpoints.db"

from streamlit.testing.v1 import AppTest


def fail(msg):
    print("FAIL:", msg)
    raise SystemExit(1)


at = AppTest.from_file(APP, default_timeout=180)
at.run()
if at.exception:
    fail(f"initial run: {[getattr(e, 'value', e) for e in at.exception]}")

# 登录 alice（登录不调用 API）
at.text_input[0].set_value("alice")
at.text_input[1].set_value("alice123")
at.button[0].click().run()
md = "\n".join(m.value for m in at.markdown)
md += "\n" + "\n".join(m.value for m in at.sidebar.markdown)
if at.exception or "🤖 银发智脑・兴趣教育智慧中台" not in md:
    fail("login failed: main header missing")

# 切到隔离用户后发起一次会调用真实 API 的对话（无效 Key -> 401）
user = f"smoke-fail-{int(time.time())}"
at.session_state["username"] = user
at.session_state["display_name"] = "Smoke"
at.session_state["chat_history"] = []
at.run()

at.chat_input[0].set_value("你好，介绍一下你自己").run()

if at.exception:
    fail(f"401 must be caught, but page raised: {[getattr(e, 'value', e) for e in at.exception]}")
errors = [e.value for e in at.error]
if not any("对话执行失败" in e for e in errors):
    fail(f"missing friendly error message, errors={errors}")
hist = at.session_state["chat_history"]
if any(h["role"] == "user" for h in hist):
    fail(f"failed user message must be rolled back, history={hist}")
print("M2-5 injection OK: 401 handled, no stack page, user message rolled back")

# 清理失败注入产生的线程记录
conn = sqlite3.connect(DB)
conn.execute("DELETE FROM checkpoints WHERE thread_id LIKE 'smoke-%'")
conn.execute("DELETE FROM writes WHERE thread_id LIKE 'smoke-%'")
conn.commit()
conn.close()
print("cleanup OK")
print("KANGYANG FAILURE INJECTION PASSED")
