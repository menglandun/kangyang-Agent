# pip install streamlit
import os
import sqlite3
import hmac
import hashlib
import logging
import threading
import io
import math
from pathlib import Path
import streamlit as st
from datetime import datetime, date
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from PIL import Image, ImageDraw, ImageFont

# ============ 银发智脑 · 体验课转化助手（第1批） ============
try:
    from yinfa_zhinao import run_conversion_assistant
except Exception:  # 缺少 crewai 等依赖时不影响主应用启动
    run_conversion_assistant = None

APP_DIR = Path(__file__).resolve().parent
EDU_DB = APP_DIR / "yinfa_edu.db"
CHECKPOINT_DB = APP_DIR / "checkpoints.db"
BOT_AVATAR_PATH = APP_DIR / "robot_avatar.png"


def _code_fingerprint() -> str:
    """源码内容指纹：代码/工具更新后自动让 get_app 缓存失效，无需手动改常量。"""
    digest = hashlib.sha256()
    for name in ("streamlit_app.py", "yinfa_zhinao.py"):
        try:
            digest.update((APP_DIR / name).read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()[:12]


APP_BUILD = _code_fingerprint()

# ============ 用户存储 ============
USERS = {
    "alice":  {"password": "alice123",  "name": "运营总监"},
    "bob":    {"password": "bob123",    "name": "教研主管"},
    "admin":  {"password": "admin123",  "name": "管理员"},
}

# ============ 配置 ============
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "yinfa-zhinao"

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY，例如：export DEEPSEEK_API_KEY=sk-xxxx")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0.1,
)

logger = logging.getLogger("yinfa_zhinao")

# ============ 业务数据库初始化 ============
def init_business_db():
    """初始化银发兴趣教育业务数据库（首次运行自动建表）。"""
    conn = sqlite3.connect(str(EDU_DB), timeout=10)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS students(
        id TEXT PRIMARY KEY, name TEXT, age INT, city TEXT,
        course TEXT, tag TEXT, note TEXT)""")

    c.execute("""CREATE TABLE IF NOT EXISTS checkin_plans(
        date TEXT, theme TEXT, course TEXT, tip TEXT)""")

    c.execute("""CREATE TABLE IF NOT EXISTS events(
        date TEXT, time TEXT, name TEXT, location TEXT, audience TEXT)""")

    # 唯一约束：防止多实例并发首启重复写入当天种子数据
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_checkin_plans_date ON checkin_plans(date)")
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_events_day ON events(date, time, name)")

    # 种子数据：学员只在首次建库时写入
    c.execute("SELECT COUNT(*) FROM students")
    if c.fetchone()[0] == 0:
        students = [
            ("S001", "王秀兰", 68, "北京", "手机摄影", "高意向", "连续打卡7天，问过进阶班价格"),
            ("S002", "张建国", 72, "上海", "书法", "观望", "上过两次体验课，还没报正课"),
            ("S003", "刘桂芳", 65, "杭州", "声乐", "高意向", "已购基础课，作业完成度高"),
            ("S004", "赵德明", 70, "广州", "手机摄影", "沉默", "两周没打开课程，需要唤醒"),
            ("S005", "孙丽华", 63, "成都", "短视频剪辑", "观望", "担心手机剪辑太难上手"),
        ]
        c.executemany("INSERT OR IGNORE INTO students VALUES(?,?,?,?,?,?,?)", students)

    # 打卡主题/课程活动按天补齐：每天首次运行写入当天数据，避免隔天查询为空
    today = date.today().isoformat()
    c.execute("SELECT COUNT(*) FROM checkin_plans WHERE date=?", (today,))
    if c.fetchone()[0] == 0:
        checkin_plans = [
            (today, "窗边的光", "手机摄影", "拍一张家里窗边的光影，发到社群完成今日打卡"),
        ]
        c.executemany("INSERT OR IGNORE INTO checkin_plans VALUES(?,?,?,?)", checkin_plans)

    c.execute("SELECT COUNT(*) FROM events WHERE date=?", (today,))
    if c.fetchone()[0] == 0:
        events = [
            (today, "14:00", "直播课：手机摄影第3课（构图进阶）", "直播间A", "手机摄影班学员"),
            (today, "19:00", "社群活动：书法作业集中点评", "学员群", "书法班学员"),
            (today, "20:00", "体验课：声乐入门公开课", "直播间B", "新学员"),
        ]
        c.executemany("INSERT OR IGNORE INTO events VALUES(?,?,?,?,?)", events)

    conn.commit()
    conn.close()

init_business_db()

# ============ 业务工具 ============
@tool
def query_student(student_name: str) -> str:
    """查询学员画像档案：年龄、城市、所学课程、意向标签和运营备注。"""
    conn = sqlite3.connect(str(EDU_DB), timeout=10)
    c = conn.cursor()
    c.execute("SELECT * FROM students WHERE name LIKE ?", (f"%{student_name}%",))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return f"未找到名为 {student_name} 的学员，可尝试查询全部学员"
    result = []
    for r in rows:
        result.append(f"档案号 {r[0]}：{r[1]}，{r[2]}岁，来自{r[3]}，课程「{r[4]}」，意向标签：{r[5]}。运营备注：{r[6]}")
    return "\n".join(result)

@tool
def list_students() -> str:
    """列出全部学员及其意向标签。"""
    conn = sqlite3.connect(str(EDU_DB), timeout=10)
    c = conn.cursor()
    c.execute("SELECT name, course, tag FROM students")
    rows = c.fetchall()
    conn.close()
    return "当前学员：\n" + "\n".join([f"- {r[0]}（{r[1]} · {r[2]}）" for r in rows])

@tool
def query_checkin_topic() -> str:
    """查询今日打卡主题与玩法提示。"""
    conn = sqlite3.connect(str(EDU_DB), timeout=10)
    c = conn.cursor()
    today = date.today().isoformat()
    c.execute("SELECT theme, course, tip FROM checkin_plans WHERE date=?", (today,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return "今日暂无打卡主题"
    theme, course, tip = rows[0]
    return f"今日打卡主题（{today}）：{theme}（{course}）\n玩法提示：{tip}"

@tool
def list_today_events() -> str:
    """查询今日课程与社群活动安排。"""
    conn = sqlite3.connect(str(EDU_DB), timeout=10)
    c = conn.cursor()
    today = date.today().isoformat()
    c.execute("SELECT time, name, location, audience FROM events WHERE date=?", (today,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return "今日暂无课程或活动安排"
    return f"今日课程/活动（{today}）：\n" + "\n".join([f"- {r[0]} {r[1]} @ {r[2]}（{r[3]}）" for r in rows])

@tool
def send_notification(target: str, title: str, content: str) -> str:
    """给学员或社群发消息（需主管审批后才真正发送）。target 可以是学员姓名或群名。"""
    return f"消息已发送至 {target}：「{title}」— {content}"

tools = [query_student, list_students, query_checkin_topic, list_today_events, send_notification]
llm_with_tools = llm.bind_tools(tools)

# ============ 图 ============
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def call_llm(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def human_approval(state: AgentState):
    """中断点：发送消息前暂停；interrupt() 会把状态存进 checkpointer。"""
    tool_calls = state["messages"][-1].tool_calls
    notice_call = next(tc for tc in tool_calls if tc["name"] == "send_notification")
    approved = interrupt({
        "action": "需要审批发消息",
        "tool": notice_call["name"],
        "args": notice_call["args"],
        "tool_calls": [{"name": tc["name"], "args": tc["args"]} for tc in tool_calls],
    })
    if approved:
        # 人工批准：整批调用交给 tools 节点执行
        return {}
    # 人工拒绝：为本批每个 tool_call 逐条补 ToolMessage，保证消息契约完整
    return {
        "messages": [
            ToolMessage(
                content=f"用户拒绝了本次操作（{tc['name']}），未发送。",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

tool_node = ToolNode(tools)

def route_after_llm(state: AgentState):
    last_msg = state["messages"][-1]
    if not last_msg.tool_calls:
        return END
    return "human_approval" if any(tc["name"] == "send_notification" for tc in last_msg.tool_calls) else "tools"

def route_after_approval(state: AgentState):
    """审批通过 -> 执行工具；被拒绝 -> 带着拒绝结果回到 LLM。"""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, ToolMessage):
        return "llm"
    return "tools"

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("human_approval", human_approval)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", route_after_llm, {
    "human_approval": "human_approval", "tools": "tools", END: END,
})
graph.add_conditional_edges("human_approval", route_after_approval, {
    "tools": "tools", "llm": "llm",
})
graph.add_edge("tools", "llm")

# ============ 登录 ============
def verify_user(username: str, password: str) -> bool:
    if username not in USERS:
        return False
    try:
        return hmac.compare_digest(
            USERS[username]["password"].encode("utf-8"), password.encode("utf-8")
        )
    except (TypeError, ValueError, AttributeError):
        return False

def login_page():
    st.set_page_config(page_title="登录 - 银发智脑・兴趣教育智慧中台", page_icon="🔐")
    st.title("🔐 银发智脑・兴趣教育智慧中台")
    st.markdown("学员运营 · 课程交付 · 社群互动 · 转化留存")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(LOGIN_HIDE_TOOLBAR_CSS, unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录", type="primary")
    if submitted:
        if verify_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["display_name"] = USERS[username]["name"]
            st.rerun()
        else:
            st.error("用户名或密码错误")
    st.caption("演示账号：alice/alice123（运营总监）、bob/bob123（教研主管）")

def logout():
    for key in ["authenticated", "username", "display_name", "chat_history",
                "yinfa_result", "yinfa_requested", "pending_prompt"]:
        st.session_state.pop(key, None)
    st.rerun()

# ============ 主界面 ============
@st.cache_resource
def get_app(build: str):
    # build 参与缓存键：代码/工具更新时改 APP_BUILD 即可让缓存失效
    # （注意：Streamlit 只把函数体内实际使用到的参数计入缓存键）
    _ = build
    # timeout：多进程/多实例共享同一 SQLite 时最多等 10 秒，避免瞬时 database is locked
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False, timeout=10)
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer), checkpointer

def clear_thread(checkpointer, thread_id: str) -> bool:
    """删除该线程在 checkpointer 里的全部记录；失败时提示并返回 False。"""
    try:
        if hasattr(checkpointer, "delete_thread"):
            checkpointer.delete_thread(thread_id)
        else:
            conn = checkpointer.conn
            with conn:
                conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        return True
    except Exception:
        logger.exception("清空对话失败")
        st.error("清空对话失败，请稍后重试。")
        return False

def history_from_messages(messages) -> list:
    """把 checkpoint 里的消息重建为页面历史（跳过工具消息与工具调用消息）。"""
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            history.append({"role": "assistant", "content": str(msg.content)})
    return history


# 同一用户多标签页并发点“批准/拒绝”时，进程内按 thread_id 串行，并二次校验中断是否仍存在
_resume_locks = {}
_resume_locks_guard = threading.Lock()


def resume_lock(thread_id: str) -> threading.Lock:
    with _resume_locks_guard:
        return _resume_locks.setdefault(thread_id, threading.Lock())


# 同一时刻只允许一个转化方案生成任务，避免多用户并发调用 crewai 时资源竞争
_yinfa_generation_lock = threading.Lock()


AVATAR_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]


def _avatar_font(size: int):
    for path in AVATAR_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


@st.cache_data(show_spinner=False)
def name_avatar(text: str, bg: str) -> bytes:
    """生成圆形姓名头像（≤3 字一行，长名字拆两行），用作聊天头像。"""
    size = 96
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((0, 0, size, size), fill=tuple(int(bg[i:i + 2], 16) for i in (1, 3, 5)))

    if len(text) <= 3:
        lines = [text]
    else:
        half = math.ceil(len(text) / 2)
        lines = [text[:half], text[half:]]

    font_size = 44
    while font_size > 10:
        font = _avatar_font(font_size)
        widest = max(d.textlength(line, font=font) for line in lines)
        if widest <= 76 and font_size * 1.15 * len(lines) <= 78:
            break
        font_size -= 2

    font = _avatar_font(font_size)
    line_h = font_size * 1.15
    y = (size - line_h * len(lines)) / 2
    for line in lines:
        bbox = d.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        d.text(((size - w) / 2 - bbox[0], y + (line_h - h) / 2 - bbox[1]), line, font=font, fill="white")
        y += line_h

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def bot_avatar_bytes() -> bytes:
    """机器人头像：读取脚本同目录的 robot_avatar.png，文件缺失时退回内置图案。"""
    try:
        return BOT_AVATAR_PATH.read_bytes()
    except Exception:
        return name_avatar("智", "#F59E0B")


@st.cache_data(show_spinner=False)
def student_names() -> list:
    """学员名单（用于侧栏“查学员画像”快捷入口）。"""
    try:
        conn = sqlite3.connect(str(EDU_DB), timeout=10)
        rows = conn.execute("SELECT name FROM students ORDER BY id").fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


APP_CSS = """
<style>
.kangyang-sidebar-title {
    font-size: 1.1rem;
    font-weight: 700;
    line-height: 1.4;
    margin: 0;
    padding: 0.15rem 0 0.35rem 0;
}
.chat-sender-name {
    font-size: 0.8rem;
    font-weight: 600;
    line-height: 1.2;
    margin: 0 0 0.25rem 0;
    opacity: 0.65;
}
.kangyang-welcome {
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.4;
    margin: 0;
    padding: 0.2rem 0 0.3rem 0;
}
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
    min-height: 100%;
    display: flex;
    flex-direction: column;
}
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] > div {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
}
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] div[data-testid="stVerticalBlock"] {
    flex: 1 1 auto;
}
div[data-testid="stElementContainer"]:has(.sidebar-bottom-anchor) {
    margin-top: auto;
}

/* 右上角工具栏中文化 */
button[data-testid="stBaseButton-header"] span { display: none; }
button[data-testid="stBaseButton-header"]::before { content: "🤖"; margin-right: 0.35rem; }
button[data-testid="stBaseButton-header"]::after { content: "设置"; }

[data-testid="stMainMenuItem-rerun"] > span,
[data-testid="stMainMenuItem-clearCache"] > span,
[data-testid="stMainMenuItem-print"] > span,
[data-testid="stMainMenuItem-recordScreencast"] > span { display: none; }

[data-testid="stMainMenuItem-rerun"]::after { content: "重新运行"; }
[data-testid="stMainMenuItem-clearCache"]::after { content: "清除缓存"; }
[data-testid="stMainMenuItem-print"]::after { content: "打印"; }
[data-testid="stMainMenuItem-recordScreencast"]::after { content: "录屏"; }

[data-testid="stMainMenuItem-autoRerun"] { font-size: 0; }
[data-testid="stMainMenuItem-autoRerun"]::before { content: "自动重新运行"; font-size: 0.875rem; }

[data-testid^="stMainMenuItem-theme-"] { font-size: 0; }
[data-testid^="stMainMenuItem-theme-"] [data-testid="stIconMaterial"] { font-size: 1.25rem; }
[data-testid="stMainMenuItem-theme-System"]::after { content: "跟随系统"; font-size: 0.875rem; }
[data-testid="stMainMenuItem-theme-Light"]::after { content: "浅色"; font-size: 0.875rem; }
[data-testid="stMainMenuItem-theme-Dark"]::after { content: "深色"; font-size: 0.875rem; }

div:has(> .stMenuVersionCopyButton) > span { font-size: 0; }
div:has(> .stMenuVersionCopyButton) > span::after { content: "由 Streamlit 提供支持"; font-size: 0.75rem; }
</style>
"""

LOGIN_HIDE_TOOLBAR_CSS = """
<style>
/* 登录页不显示右上角工具栏（设置按钮 + 主菜单），登录后在主界面显示 */
button[data-testid="stBaseButton-header"] { display: none; }
button[data-testid="stMainMenuButton"] { display: none; }
</style>
"""

def main_app():
    st.set_page_config(page_title="银发智脑・兴趣教育智慧中台", page_icon="🤖", layout="wide")

    # 仅隐藏云端注入的额外按钮（Star/Fork 等）与右下角 Manage app，保留“设置”和 ⋮ 菜单
    st.markdown(
        """
<style>
/* 云端平台注入的操作按钮（Star/Edit/Fork 等），不影响“设置”和主菜单 */
[data-testid="stToolbarActions"] { display: none; }
/* 右下角 Manage app 按钮（本地对应运行状态条） */
[data-testid="stStatusWidget"] { display: none; }
</style>
""",
        unsafe_allow_html=True,
    )

    display_name = st.session_state["display_name"]
    user_id = st.session_state["username"]

    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown('<div class="kangyang-welcome">欢迎来到兴趣教育家园</div>', unsafe_allow_html=True)
    st.caption(f"当前登录：{display_name}（@{user_id}）")
    st.caption("工作流：获客 → 转化 → 交付 → 运营 → 留存 ｜ 第1批已上线：体验课7天转化助手（学员画像 · 社群话术 · 作业点评 · 转化建议）")

    app, checkpointer = get_app(APP_BUILD)
    config = {"configurable": {"thread_id": user_id}}

    user_avatar = name_avatar(display_name, "#3B82F6")
    bot_avatar = bot_avatar_bytes()

    snapshot = app.get_state(config)
    pending = bool(snapshot and snapshot.tasks and snapshot.tasks[0].interrupts)

    def queue_prompt(text: str):
        st.session_state["pending_prompt"] = text
        st.rerun()

    with st.sidebar:
        st.markdown('<div class="kangyang-sidebar-title">🤖 银发智脑・兴趣教育智慧中台</div>', unsafe_allow_html=True)
        st.divider()
        if st.button("🆕 新建对话", use_container_width=True):
            if clear_thread(checkpointer, user_id):
                st.session_state.pop("chat_history", None)
                st.rerun()
        st.divider()
        st.markdown("**可用功能**")
        with st.popover("👤 查学员画像", disabled=pending, use_container_width=True):
            for name in student_names():
                if st.button(name, key=f"student-{name}"):
                    queue_prompt(f"查询学员{name}的画像")
        if st.button("📋 列全部学员", use_container_width=True, disabled=pending):
            queue_prompt("列出全部学员和意向标签")
        if st.button("📅 今日打卡主题", use_container_width=True, disabled=pending):
            queue_prompt("今天的打卡主题是什么？")
        if st.button("🎯 今日课程安排", use_container_width=True, disabled=pending):
            queue_prompt("今天有哪些课程和活动安排？")
        with st.popover("📢 发消息（需审批）", disabled=pending, use_container_width=True):
            target = st.text_input("接收人/群", placeholder="如：王秀兰 / 书法班群", key="notice-target")
            title = st.text_input("标题", placeholder="如：打卡提醒", key="notice-title")
            content = st.text_area("内容", placeholder="如：今晚 19:00 有社群答疑，记得参加", key="notice-content")
            if st.button("提交消息", key="notice-submit"):
                if target.strip() and title.strip() and content.strip():
                    queue_prompt(f"给{target.strip()}发一条消息，标题：{title.strip()}，内容：{content.strip()}")
                else:
                    st.caption("请填写接收人、标题和内容")
        st.divider()
        st.markdown("**转化助手**")
        with st.popover("🎓 体验课7天转化助手", disabled=pending, use_container_width=True):
            st.caption("示例学员：张阿姨（62岁 · 书法体验课 · 第5天）。4个智能体依次生成：学员画像 → 社群话术 → 作业点评 → 转化建议，约需 1-2 分钟。")
            if run_conversion_assistant is None:
                st.caption("当前环境未安装 crewai，功能暂不可用")
            elif st.button("🚀 生成今日转化方案", key="yinfa-run", type="primary", use_container_width=True):
                st.session_state["yinfa_requested"] = True
                st.rerun()
        st.markdown('<div class="sidebar-bottom-anchor"></div>', unsafe_allow_html=True)
        if st.button("🚪 退出登录"):
            logout()

    # 对话历史（会话内没有历史时，从该线程的 checkpoint 重建）
    if "chat_history" not in st.session_state:
        saved_messages = (app.get_state(config).values or {}).get("messages", [])
        st.session_state.chat_history = history_from_messages(saved_messages)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user", avatar=user_avatar):
                st.markdown(f'<div class="chat-sender-name">@{user_id}</div>', unsafe_allow_html=True)
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar=bot_avatar):
                st.markdown('<div class="chat-sender-name">大聪明智能助手</div>', unsafe_allow_html=True)
                st.markdown(msg["content"])

    # 审批（挂起期间禁用输入，避免新输入打断或覆盖待审批流程）
    if pending:
        interrupt_data = snapshot.tasks[0].interrupts[0].value
        st.warning(f"⚠️ 待审批：{interrupt_data}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 批准发送", type="primary"):
                with resume_lock(user_id):
                    fresh = app.get_state(config)
                    if not (fresh.tasks and fresh.tasks[0].interrupts):
                        st.rerun()  # 已被其他会话处理，避免重复 resume
                    try:
                        result = app.invoke(Command(resume=True), config)
                    except Exception:
                        logger.exception("审批处理失败（批准）")
                        st.error("审批处理失败，请刷新页面后重试。")
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "content": result["messages"][-1].content})
                        st.rerun()
        with col2:
            if st.button("❌ 拒绝"):
                with resume_lock(user_id):
                    fresh = app.get_state(config)
                    if not (fresh.tasks and fresh.tasks[0].interrupts):
                        st.rerun()  # 已被其他会话处理，避免重复 resume
                    try:
                        result = app.invoke(Command(resume=False), config)
                    except Exception:
                        logger.exception("审批处理失败（拒绝）")
                        st.error("审批处理失败，请刷新页面后重试。")
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "content": result["messages"][-1].content})
                        st.rerun()

    # 体验课转化助手（第1批）：按需生成，结果保留在会话里
    if st.session_state.pop("yinfa_requested", False):
        if run_conversion_assistant is None:
            st.error("当前环境缺少 crewai 依赖，无法生成转化方案。")
        elif not _yinfa_generation_lock.acquire(blocking=False):
            st.warning("已有生成任务正在进行，请稍后再试。")
        else:
            try:
                with st.spinner("银发智脑正在生成今日转化方案（4个智能体依次工作，约需 1-2 分钟）..."):
                    st.session_state["yinfa_result"] = run_conversion_assistant()
            except Exception:
                logger.exception("转化方案生成失败")
                st.session_state.pop("yinfa_result", None)
                st.error("转化方案生成失败，请稍后重试。")
            finally:
                _yinfa_generation_lock.release()

    yinfa_result = st.session_state.get("yinfa_result")
    if yinfa_result:
        st.divider()
        st.subheader("🎓 体验课7天转化助手 · 今日方案")
        st.caption("示例学员：张阿姨（62岁 · 书法体验课 · 第5天/共7天）")
        for section_name, section_text in yinfa_result.items():
            with st.expander(section_name, expanded=(section_name == "转化建议")):
                st.markdown(section_text or "（暂无输出）")

    # 输入（待审批解决前不接受新的普通输入；侧栏快捷功能通过 pending_prompt 注入）
    typed = st.chat_input("例如：查一下王秀兰的学员画像？今天的打卡主题是什么？", disabled=pending)
    queued = st.session_state.pop("pending_prompt", None) if not pending else None
    prompt = typed or queued
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(f'<div class="chat-sender-name">@{user_id}</div>', unsafe_allow_html=True)
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=bot_avatar):
            st.markdown('<div class="chat-sender-name">大聪明智能助手</div>', unsafe_allow_html=True)
            try:
                result = app.invoke({"messages": [("user", prompt)]}, config)
            except Exception:
                logger.exception("对话执行失败")
                st.session_state.chat_history.pop()
                st.error("对话执行失败，请重试。")
            else:
                snapshot = app.get_state(config)
                if snapshot.tasks and snapshot.tasks[0].interrupts:
                    st.info("⏸️ 等待审批...")
                    st.rerun()
                else:
                    answer = result["messages"][-1].content
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ============ 路由 ============
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_page()
else:
    main_app()
