"""E2E regression for the kangyang streamlit_app.py via Streamlit AppTest (real DeepSeek API)."""
import os
import sqlite3
import time
from pathlib import Path

DIR = os.environ.get("KANGYANG_APP_DIR") or str(Path(__file__).resolve().parents[1])
APP = f"{DIR}/streamlit_app.py"
DB = f"{DIR}/checkpoints.db"

from streamlit.testing.v1 import AppTest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def fail(msg):
    print("FAIL:", msg)
    raise SystemExit(1)


def markdown_text():
    return "\n".join(m.value for m in at.markdown)


def all_text():
    return markdown_text() + "\n" + "\n".join(m.value for m in at.sidebar.markdown)


at = AppTest.from_file(APP, default_timeout=180)
at.run()
if at.exception:
    fail(f"initial run: {[getattr(e, 'value', e) for e in at.exception]}")
if at.title[0].value != "🔐 银发智脑・兴趣教育智慧中台":
    fail(f"expected login page, got title: {at.title[0].value}")
print("login page renders OK")

# --- 错误密码 / 非 ASCII 密码 ---
at.text_input[0].set_value("alice")
at.text_input[1].set_value("wrong-password")
at.button[0].click().run()
if at.exception:
    fail(f"bad login run: {[getattr(e, 'value', e) for e in at.exception]}")
if not any("用户名或密码错误" in e.value for e in at.error):
    fail(f"wrong-password error missing: {[e.value for e in at.error]}")

at.text_input[1].set_value("密码123")
at.button[0].click().run()
if at.exception:
    fail(f"non-ascii password run: {[getattr(e, 'value', e) for e in at.exception]}")
if not any("用户名或密码错误" in e.value for e in at.error):
    fail(f"non-ascii password should be rejected gracefully: {[e.value for e in at.error]}")
print("password checks OK (wrong + non-ascii, no crash)")

# --- 正确登录 ---
at.text_input[1].set_value("alice123")
at.button[0].click().run()
if at.exception:
    fail(f"login run: {[getattr(e, 'value', e) for e in at.exception]}")
if "🤖 银发智脑・兴趣教育智慧中台" not in all_text():
    fail(f"expected main app header, markdown={all_text()[:200]}")
if not at.session_state["authenticated"] or at.session_state["username"] != "alice":
    fail("session not authenticated as alice")
print("login as alice OK")

# --- 隔离测试用户 ---
user = f"smoke-{int(time.time())}"
at.session_state["username"] = user
at.session_state["display_name"] = "Smoke"
at.session_state["chat_history"] = []
at.run()
if at.exception:
    fail(f"switch user run: {[getattr(e, 'value', e) for e in at.exception]}")

# --- 学员画像查询 ---
at.chat_input[0].set_value("查一下王秀兰的学员画像").run()
if at.exception:
    fail(f"student query: {[getattr(e, 'value', e) for e in at.exception]}")
if "高意向" not in markdown_text():
    fail("student answer missing 高意向")
print("student query OK")

# --- 侧栏快捷功能（可点击） ---
quick = None
for b in at.button:
    if "今日打卡主题" in b.label:
        quick = b
        break
if quick is None:
    fail(f"quick action button missing: {[b.label for b in at.button]}")
quick.click().run()
if at.exception:
    fail(f"quick action run: {[getattr(e, 'value', e) for e in at.exception]}")
if "窗边" not in markdown_text():
    fail("quick action answer missing 窗边")
print("sidebar quick action OK")

# --- 发消息 -> 审批 -> 执行 ---
at.chat_input[0].set_value("帮我给王秀兰发一条消息，标题：打卡提醒，内容：今晚记得完成摄影打卡。").run()
if at.exception:
    fail(f"notification run: {[getattr(e, 'value', e) for e in at.exception]}")
if not any("批准发送" in b.label for b in at.button):
    fail(f"approval buttons missing: {[b.label for b in at.button]}")
chat_disabled = getattr(at.chat_input[0], "disabled", None)
if chat_disabled is False:
    fail("chat_input should be disabled while approval is pending")
print("notification pending OK" + ("" if chat_disabled is None else f" (disabled={chat_disabled})"))

for b in at.button:
    if "批准发送" in b.label:
        b.click().run()
        break
if at.exception:
    fail(f"approve run: {[getattr(e, 'value', e) for e in at.exception]}")
if any("批准发送" in b.label for b in at.button):
    fail("approval UI still pending after approve")

hist = at.session_state["chat_history"]
if len(hist) != 6 or hist[-1]["role"] != "assistant":
    fail(f"chat history unexpected: {[(h['role'], h['content'][:40]) for h in hist]}")

conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
row = conn.execute(
    "select type, checkpoint from checkpoints where thread_id=? order by rowid desc limit 1", (user,)
).fetchone()
conn.close()
ckpt = JsonPlusSerializer().loads_typed((row[0], row[1]))
msgs = ckpt.get("channel_values", {}).get("messages", [])
if not any(type(m).__name__ == "ToolMessage" and "消息已发送至" in str(m.content) for m in msgs):
    fail("send_notification ToolMessage not found in saved state")
print("approve & resume OK (message executed)")

# --- 清空当前对话 ---
cleared = False
for b in at.button:
    if "新建对话" in b.label:
        b.click().run()
        cleared = True
        break
if not cleared:
    fail("new conversation button missing")
if at.exception:
    fail(f"clear run: {[getattr(e, 'value', e) for e in at.exception]}")
if at.session_state["chat_history"] != []:
    fail(f"clear should empty page history, got {at.session_state['chat_history']}")
conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
left = 0
for table in ("checkpoints", "writes"):
    left += conn.execute(f"select count(*) from {table} where thread_id=?", (user,)).fetchone()[0]
conn.close()
if left != 0:
    fail(f"clear should delete thread checkpoints, {left} rows left")
print("clear conversation OK (page + checkpoint thread cleared)")

# --- 退出登录 ---
for b in at.button:
    if "退出登录" in b.label:
        b.click().run()
        break
if at.exception:
    fail(f"logout run: {[getattr(e, 'value', e) for e in at.exception]}")
if at.title[0].value != "🔐 银发智脑・兴趣教育智慧中台" or at.session_state["authenticated"]:
    fail("logout did not return to login page")
print("logout OK")

# --- 清理测试线程 ---
conn = sqlite3.connect(DB)
conn.execute("DELETE FROM checkpoints WHERE thread_id LIKE 'smoke-%'")
conn.execute("DELETE FROM writes WHERE thread_id LIKE 'smoke-%'")
conn.commit()
conn.close()
print("cleanup OK")

print("KANGYANG E2E PASSED")
