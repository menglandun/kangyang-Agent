"""银发智脑 · 每周打卡主题生成（独立模块）。

- 独立运行：python yinfa_zhinao_checkin.py（先在 .env 或环境变量里配置 DEEPSEEK_API_KEY）
- 应用集成：from yinfa_zhinao_checkin import run_checkin_plan
"""
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from yinfa_zhinao_llm import build_deepseek_llm

load_dotenv()

# ---------- 模拟输入数据 ----------
DEFAULT_WEEK_INFO = """当前时间：2026年10月第三周
学员类型：书法体验课学员（60-70岁退休人群）
本周节点：重阳节（10月19日）
班级人数：150人
"""


def run_checkin_plan(week_info=None):
    """运行打卡主题策划员，返回分节结果。"""
    week_info = week_info or DEFAULT_WEEK_INFO

    deepseek_llm = build_deepseek_llm()

    planner = Agent(
        role="社群打卡主题策划员",
        goal="根据当前时间节点和学员特点，生成一周7天的社群打卡主题",
        backstory="""你是中老年社群运营专家，最懂怎么让60后阿姨大叔愿意主动打卡。
    你设计的主题接地气、有共鸣、不尴尬，让学员愿意晒、愿意聊。""",
        verbose=True,
        allow_delegation=False,
        llm=deepseek_llm,
    )
    task = Task(
        description=f"""根据以下信息，生成本周7天的社群打卡主题：
    {week_info}

    要求：
    1. 每天一个主题，从周一到周日
    2. 主题要跟书法学习或退休生活相关
    3. 配合重阳节节点设计1-2个特别主题
    4. 每天附一段引导话术（班主任直接发群里）""",
        expected_output="7天打卡主题表，每天含主题名+引导话术。",
        agent=planner,
    )

    crew = Crew(agents=[planner], tasks=[task], process=Process.sequential, verbose=True)
    final = crew.kickoff()

    raw = getattr(getattr(task, "output", None), "raw", None)
    text = str(raw).strip() if raw else str(final).strip()
    return {"7天打卡主题表": text}


if __name__ == "__main__":
    sections = run_checkin_plan()
    print("\n" + "=" * 60)
    print("银发智脑 · 每周打卡主题 最终输出")
    print("=" * 60)
    for name, text in sections.items():
        print(f"\n【{name}】\n{text}")
