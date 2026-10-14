"""银发智脑 · 续费预警与挽留（独立模块）。

- 独立运行：python yinfa_zhinao_renewal.py（先在 .env 或环境变量里配置 DEEPSEEK_API_KEY）
- 应用集成：from yinfa_zhinao_renewal import run_renewal_analysis
"""
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from yinfa_zhinao_llm import build_deepseek_llm

load_dotenv()

# ---------- 模拟输入数据 ----------
DEFAULT_RISK_DATA = """学员：李阿姨，65岁，正在学行书正价课
课程进度：已学2/4个月，还剩2个月到期
近7天登录：第1天30分钟，第2天10分钟，第3-5天完全没登录
作业：最近3次只交了1次，质量一般
社群：近7天没发言，之前每周发言2-3次
班主任联系：昨天发了微信，没回
续费意向：之前提过“学完这期看看”
投诉记录：无
"""


def run_renewal_analysis(risk_data=None):
    """运行流失预警员，返回分节结果。"""
    risk_data = risk_data or DEFAULT_RISK_DATA

    deepseek_llm = build_deepseek_llm()

    detector = Agent(
        role="学员流失预警员",
        goal="根据学员行为数据，判断流失风险等级，并生成挽留建议",
        backstory="""你是中老年教育行业的用户运营专家，带过几万人的学员社群。
    你最擅长从蛛丝马迹里看出谁快要流失了，并且能用最温暖的方式把人留住。""",
        verbose=True,
        allow_delegation=False,
        llm=deepseek_llm,
    )
    task = Task(
        description=f"""分析以下学员行为数据，输出：
    1. 流失风险等级（高/中/低）
    2. 判断依据（从哪些信号看出来的）
    3. 挽留建议（现在该做什么）
    4. 一段私聊挽留话术（班主任直接发）

    学员数据：
    {risk_data}""",
        expected_output="风险等级+依据+挽留建议+一段可直接发的私聊话术。",
        agent=detector,
    )

    crew = Crew(agents=[detector], tasks=[task], process=Process.sequential, verbose=True)
    final = crew.kickoff()

    raw = getattr(getattr(task, "output", None), "raw", None)
    text = str(raw).strip() if raw else str(final).strip()
    return {"流失风险分析与挽留话术": text}


if __name__ == "__main__":
    sections = run_renewal_analysis()
    print("\n" + "=" * 60)
    print("银发智脑 · 续费预警与挽留 最终输出")
    print("=" * 60)
    for name, text in sections.items():
        print(f"\n【{name}】\n{text}")
