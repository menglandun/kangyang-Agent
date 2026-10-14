"""银发智脑 · 作业批改（独立模块）。

- 独立运行：python yinfa_zhinao_grading.py（先在 .env 或环境变量里配置 DEEPSEEK_API_KEY）
- 应用集成：from yinfa_zhinao_grading import run_grading
"""
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from yinfa_zhinao_llm import build_deepseek_llm

load_dotenv()

# ---------- 模拟输入数据 ----------
DEFAULT_HOMEWORK = """学员：王大叔，68岁，行书班
作业内容：临摹《兰亭序》“永和九年”四个字
自评：觉得写得有点歪，不满意
"""


def run_grading(homework=None):
    """运行作业批改老师，返回分节结果。"""
    homework = homework or DEFAULT_HOMEWORK

    deepseek_llm = build_deepseek_llm()

    grader = Agent(
        role="作业批改老师",
        goal="批改学员作业，给出分数和具体改进建议",
        backstory="""你是教了20年中老年学员的书法老师，批改作业严格但温暖。
    你先肯定优点，再指出1-2个具体可以改进的地方，最后鼓励。""",
        verbose=True,
        allow_delegation=False,
        llm=deepseek_llm,
    )
    task = Task(
        description=f"""批改以下学员作业：
    {homework}

    要求输出：
    1. 先夸（具体夸哪里）
    2. 再提1-2个具体改进建议（要具体，不要"写好一点"这种空话）
    3. 最后鼓励
    4. 给一个分数（10分制）""",
        expected_output="一段100字左右的作业批改+分数。",
        agent=grader,
    )

    crew = Crew(agents=[grader], tasks=[task], process=Process.sequential, verbose=True)
    final = crew.kickoff()

    raw = getattr(getattr(task, "output", None), "raw", None)
    text = str(raw).strip() if raw else str(final).strip()
    return {"作业批改": text}


if __name__ == "__main__":
    sections = run_grading()
    print("\n" + "=" * 60)
    print("银发智脑 · 作业批改 最终输出")
    print("=" * 60)
    for name, text in sections.items():
        print(f"\n【{name}】\n{text}")
