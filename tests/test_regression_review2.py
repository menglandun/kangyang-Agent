"""Regression tests锁定第二轮评审发现的缺陷（未修复前应为红色）。

- BUG-R2a: logout 未清理 yinfa_result → 换用户登录会看到上一位用户的转化方案
- BUG-R2b: 种子表缺少唯一约束 → 多实例并发首启会重复写入当天数据
- BUG-R2c: requirements 未锁版本 → 云端自动重部署存在上游升级导致的构建风险

Run: PYTHONDONTWRITEBYTECODE=1 python3 work/test_regression_review2.py
"""
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import date
from pathlib import Path

os.environ["DEEPSEEK_API_KEY"] = "test-key"

DIR = os.environ.get("KANGYANG_APP_DIR") or str(Path(__file__).resolve().parents[1])
APP = f"{DIR}/streamlit_app.py"
EDU_DB = f"{DIR}/yinfa_edu.db"
REQ = f"{DIR}/requirements.txt"

from streamlit.testing.v1 import AppTest


def text(at):
    parts = [m.value for m in at.markdown]
    for attr in ("subheader", "header", "title"):
        try:
            parts += [e.value for e in getattr(at, attr)]
        except Exception:
            pass
    return "\n".join(parts)


def login(at, username, password):
    at.run()
    at.text_input[0].set_value(username)
    at.text_input[1].set_value(password)
    at.button[0].click().run()


def click_button(at, label_part):
    for b in at.button:
        if label_part in b.label:
            b.click().run()
            return True
    return False


def test_logout_clears_conversion_result():
    """BUG-R2a: 退出登录必须清掉转换助手产物，避免跨用户残留展示。"""
    at = AppTest.from_file(APP, default_timeout=120)
    login(at, "alice", "alice123")
    at.session_state["yinfa_result"] = {"学员画像": "测试残留方案"}
    at.run()
    assert "测试残留方案" in text(at), "前置条件不成立：转化方案面板未展示"

    assert click_button(at, "退出登录"), "未找到退出登录按钮"
    assert at.title[0].value.startswith("🔐"), "退出后未回到登录页"

    leaked = False
    try:
        at.session_state["yinfa_result"]
        leaked = True
    except KeyError:
        pass
    if leaked:
        login(at, "bob", "bob123")
        visible = "测试残留方案" in text(at)
        raise AssertionError(f"BUG-R2a: logout 未清理 yinfa_result（bob 登录后仍可见: {visible}）")
    print("R2a OK: logout 已清理转化助手产物")


def test_seed_rows_are_unique_per_day():
    """BUG-R2b: 并发重复写入同一天种子数据时应保持单行（需要唯一约束）。"""
    tmp = tempfile.mkdtemp(prefix="r2b-")
    try:
        db = os.path.join(tmp, "copy.db")
        shutil.copy(EDU_DB, db)
        conn = sqlite3.connect(db)
        today = date.today().isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO checkin_plans VALUES(?,?,?,?)",
            (today, "窗边的光", "手机摄影", "并发重复写入"),
        )
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM checkin_plans WHERE date=?", (today,)).fetchone()[0]
        conn.close()
        assert n == 1, f"BUG-R2b: 同日打卡主题出现 {n} 条（缺唯一约束，并发首启会重复种子）"
        print("R2b OK: 同一天种子数据保持唯一")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_requirements_are_pinned():
    """BUG-R2c: 依赖应锁定兼容版本，保证云端构建可复现。"""
    unpinned = []
    for line in open(REQ, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not any(sym in line for sym in ("==", "~=", ">=", "<=", "!=", ">", "<")):
            unpinned.append(line)
    assert not unpinned, f"BUG-R2c: 未锁版本的依赖: {unpinned}"
    print("R2c OK: requirements 全部锁定版本")


def main():
    failures = []
    for fn in (test_logout_clears_conversion_result, test_seed_rows_are_unique_per_day, test_requirements_are_pinned):
        try:
            fn()
        except AssertionError as exc:
            failures.append(str(exc))
            print(f"FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{fn.__name__} 执行异常: {type(exc).__name__}: {exc}")
            print(f"ERROR: {fn.__name__} -> {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"REVIEW-2 REGRESSION: {len(failures)} 项未通过（即评审确认、待修复的缺陷）")
        sys.exit(1)
    print("REVIEW-2 REGRESSION PASSED")


if __name__ == "__main__":
    main()
