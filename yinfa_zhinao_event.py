"""银发智脑 · 节日活动策划（独立模块）。

- 独立运行：python yinfa_zhinao_event.py（先在 .env 或环境变量里配置 DEEPSEEK_API_KEY）
- 应用集成：from yinfa_zhinao_event import run_event_plan
"""
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from yinfa_zhinao_llm import build_deepseek_llm

load_dotenv()

# ---------- 模拟输入数据 ----------
DEFAULT_EVENT_INFO = """活动节点：重阳节（2026年10月19日（周一））
学员类型：书法正价课学员（60-70岁退休人群）
班级人数：150人
活动形式：线上社群活动
活动周期：1周（10月13日-19日）
"""


def run_event_plan(event_info=None):
    """运行活动策划师，返回分节结果。"""
    event_info = event_info or DEFAULT_EVENT_INFO

    deepseek_llm = build_deepseek_llm()

    planner = Agent(
        role="社群活动策划师",
        goal="根据节日节点和学员特点，生成完整的社群活动方案",
        backstory="""你是中老年社群活动策划专家，办过几百场线上活动。
    你设计的活动参与门槛低、氛围温暖、能让阿姨大叔们愿意晒、愿意聊。""",
        verbose=True,
        allow_delegation=False,
        llm=deepseek_llm,
    )
    task = Task(
        description=f"""根据以下信息，生成一份完整的活动方案：
    {event_info}

    要求输出：
    1. 活动主题（一句话，温暖有共鸣）
    2. 活动流程（从预热到收尾，按天排）
    3. 每天的社群文案（班主任直接发群里）
    4. 互动设计（怎么让学员愿意晒作品）
    5. 奖品建议（低成本、有仪式感）
    6. 收尾怎么做（把活动成果做成合辑）""",
        expected_output="一份完整的活动方案，可直接执行。",
        agent=planner,
    )

    crew = Crew(agents=[planner], tasks=[task], process=Process.sequential, verbose=True)
    final = crew.kickoff()

    raw = getattr(getattr(task, "output", None), "raw", None)
    text = str(raw).strip() if raw else str(final).strip()
    return {"节日活动方案": text}


if __name__ == "__main__":
    sections = run_event_plan()
    print("\n" + "=" * 60)
    print("银发智脑 · 节日活动策划 最终输出")
    print("=" * 60)
    for name, text in sections.items():
        print(f"\n【{name}】\n{text}")
