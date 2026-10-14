"""银发智脑 · 转介绍话术（独立模块）。

- 独立运行：python yinfa_zhinao_referral.py（先在 .env 或环境变量里配置 DEEPSEEK_API_KEY）
- 应用集成：from yinfa_zhinao_referral import run_referral_copy
"""
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from yinfa_zhinao_llm import build_deepseek_llm

load_dotenv()

# ---------- 模拟输入数据 ----------
DEFAULT_STUDENT_PROFILE = """学员：李阿姨，65岁，学书法3个月
特点：进步很大，很爱分享，经常在朋友圈晒作业
目标：让李阿姨推荐她的老姐妹一起来学
"""


def run_referral_copy(student_profile=None):
    """运行转介绍话术员，返回分节结果。"""
    student_profile = student_profile or DEFAULT_STUDENT_PROFILE

    deepseek_llm = build_deepseek_llm()

    writer = Agent(
        role="转介绍话术员",
        goal="生成老带新分享文案，让老学员愿意推荐新朋友",
        backstory="""你是中老年社群运营专家，最懂怎么让老学员心甘情愿地推荐新朋友。
    你写的文案不生硬、不营销，像老学员自己在朋友圈发的真心话。""",
        verbose=True,
        allow_delegation=False,
        llm=deepseek_llm,
    )
    task = Task(
        description=f"""根据以下学员特点，生成3条转介绍分享文案：
    {student_profile}

    要求：
    1. 像李阿姨自己在朋友圈发的话，不像广告
    2. 语气真诚，有真情实感
    3. 每条50-100字
    4. 包含"一起学"的邀请，但不硬推""",
        expected_output="3条可直接发朋友圈的转介绍文案。",
        agent=writer,
    )

    crew = Crew(agents=[writer], tasks=[task], process=Process.sequential, verbose=True)
    final = crew.kickoff()

    raw = getattr(getattr(task, "output", None), "raw", None)
    text = str(raw).strip() if raw else str(final).strip()
    return {"转介绍朋友圈文案": text}


if __name__ == "__main__":
    sections = run_referral_copy()
    print("\n" + "=" * 60)
    print("银发智脑 · 转介绍话术 最终输出")
    print("=" * 60)
    for name, text in sections.items():
        print(f"\n【{name}】\n{text}")
