"""银发智脑 · 课程大纲生成（独立模块）。

- 独立运行：python yinfa_zhinao_curriculum.py（先在 .env 或环境变量里配置 DEEPSEEK_API_KEY）
- 应用集成：from yinfa_zhinao_curriculum import run_curriculum
"""
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from yinfa_zhinao_llm import build_deepseek_llm

load_dotenv()

# ---------- 模拟输入数据 ----------
DEFAULT_COURSE_INFO = """课程主题：行书入门
类型：正价课
时长：2个月
学员水平：有楷书基础，刚接触行书
课时安排：每周3次课，每次45分钟，共24次课
"""


def run_curriculum(course_info=None):
    """运行课程大纲设计师，返回分节结果。"""
    course_info = course_info or DEFAULT_COURSE_INFO

    deepseek_llm = build_deepseek_llm()

    designer = Agent(
        role="课程大纲设计师",
        goal="根据课程主题和学员水平，生成完整的教学大纲和知识点拆解",
        backstory="""你是中老年兴趣教育的教研专家，设计过几十门书法、国画、智能手机课程。
    你最懂怎么把复杂的知识拆成老人能听懂、愿意学的小步骤。""",
        verbose=True,
        allow_delegation=False,
        llm=deepseek_llm,
    )
    task = Task(
        description=f"""根据以下信息，生成完整的课程大纲：
    {course_info}

    要求输出：
    1. 课程总目标（学完能达到什么水平）
    2. 按周拆解教学计划（第1周学什么、第2周学什么...）
    3. 每次课的知识点拆解（今天学什么、练什么）
    4. 作业布置建议""",
        expected_output="一份完整的课程大纲，可直接给教研团队用。",
        agent=designer,
    )

    crew = Crew(agents=[designer], tasks=[task], process=Process.sequential, verbose=True)
    final = crew.kickoff()

    raw = getattr(getattr(task, "output", None), "raw", None)
    text = str(raw).strip() if raw else str(final).strip()
    return {"课程大纲": text}


if __name__ == "__main__":
    sections = run_curriculum()
    print("\n" + "=" * 60)
    print("银发智脑 · 课程大纲生成 最终输出")
    print("=" * 60)
    for name, text in sections.items():
        print(f"\n【{name}】\n{text}")
