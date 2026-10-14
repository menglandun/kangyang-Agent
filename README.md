# kangyang-Agent

银发智脑・兴趣教育智慧中台：基于 Streamlit + LangGraph + DeepSeek 的学员运营智能体。

- 五大功能：查学员画像 / 列全部学员 / 今日打卡主题 / 今日课程安排 / 发消息（需审批）
- 体验课 7 天转化助手：4 个 crewai 智能体生成学员画像、社群话术、作业点评与转化建议
- 爆款选题分析：2 个 crewai 智能体生成爆款规律与本周选题清单、标题与钩子优化
- 运营工具集（7 个）：每周打卡主题 / 节日活动策划 / 转介绍话术 / FAQ 自动应答 / 作业批改 / 续费预警与挽留 / 课程大纲生成
- 多会话与记录：「新建对话」打开全新空窗口（同时清空窗口内工具产出）；发出首条消息即生成一条历史记录，侧栏「会话记录」可随时切回
- 提示样式：操作反馈统一为居中提示条（成功绿色，预警与报错红色，约 2 秒后自动淡出）
- 人工审批：发消息前中断等待批准，SQLite 持久化，多标签页 resume 安全

线上地址：https://kangyang-agent-5gnaym3looszturjqiyrca.streamlit.app/

## 维护说明

- 本仓库的 `yinfa_zhinao*.py` 智能体模块与「银发智脑」项目（`Agent项目：银发智脑·兴趣教育智慧中台/yinfa_zhinao`）中的 `agents/*.py` 是双份维护：调整提示词、任务描述或样例数据时需要两边同步。

## 本地运行

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-xxxx
streamlit run streamlit_app.py
```

演示账号：alice / alice123（运营总监）、bob / bob123（教研主管）、admin / admin123

## 测试

```bash
python tests/test_regression_review2.py           # 评审缺陷防复发（离线）
python tests/test_kangyang_offline.py             # 工具与审批门（离线）
python tests/test_kangyang_residual.py            # 并发 resume 与 SQLite 竞争（离线）
python tests/test_kangyang_failure_injection.py   # API 故障兜底（需网络）
DEEPSEEK_API_KEY=... python tests/test_kangyang_e2e.py              # 全链路（真实 API）
DEEPSEEK_API_KEY=... python tests/test_kangyang_refresh_history.py  # 刷新重建（真实 API）
```

GitHub Actions 在每次 push / PR 时自动运行离线套件；网络套件可在 Actions 页面手动触发（真实 API 套件需要仓库 Secret `DEEPSEEK_API_KEY`）。

测试默认以本仓库为被测项目；如有需要可用环境变量 `KANGYANG_APP_DIR` 指向其它目录。
