"""银发智脑 · 兴趣教育智慧中台 —— 第1批：体验课7天转化助手（4个智能体）。

- 独立运行：python yinfa_zhinao.py（先在 .env 或环境变量里配置 DEEPSEEK_API_KEY）
- 应用集成：from yinfa_zhinao import run_conversion_assistant
"""
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()

# ---------- 模拟输入数据 ----------
STUDENT_DATA = """学员：张阿姨，62岁，报名书法体验课
报名信息：退休教师，喜欢写字，之前在家自己练过3年
第1天：打卡（✅），发言（3次），社群活跃（高）
第2天：打卡（✅），提交作业（临摹楷书），班主任点评（已回）
第3天：打卡（❌），未发言，登录时长（5分钟）
第4天：打卡（✅），提交作业（临摹行书），说“老师写得真好，我还是写不好”
第5天：打卡（✅），发言（1次），问了一句“正价课多少钱”
课程进度：第5天/共7天
"""


def run_conversion_assistant(student_data=None):
    """运行4个智能体，返回各环节输出（学员画像/社群话术/作业点评/转化建议）。"""
    student_data = student_data or STUDENT_DATA

    deepseek_llm = LLM(
        model="deepseek/deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 密钥放在环境变量或 .env 中，不要写死在代码里
        temperature=0.7,
    )

    # ---------- 智能体1：学员画像分析员 ----------
    analyst = Agent(
        role="学员画像分析员",
        goal="根据学员7天行为数据，判断学员类型和意向程度",
        backstory="""你是中老年教育行业的资深班主任，带过几千个体验课学员。
    你一眼就能看出这个学员是高意向、观望、沉默还是快流失了。""",
        verbose=True,
        allow_delegation=False,
        llm=deepseek_llm,
    )
    task1 = Task(
        description=f"""分析以下学员数据，输出：
    1. 学员类型（高意向/观望/沉默/流失风险）
    2. 核心优势（学员的亮点是什么）
    3. 顾虑点（学员可能犹豫什么）
    4. 今天该重点关注的事

    学员数据：
    {student_data}""",
        expected_output="一份结构化的学员画像，200字以内。",
        agent=analyst,
    )

    # ---------- 智能体2：社群话术生成员 ----------
    speaker = Agent(
        role="社群话术生成员",
        goal="根据学员画像，生成今天的社群互动话术",
        backstory="""你是中老年社群运营专家，最懂怎么跟60后阿姨大叔聊天。
    你的语气温暖、亲切、像邻居一样，不浮夸、不生硬。""",
        verbose=True,
        allow_delegation=False,
        llm=deepseek_llm,
    )
    task2 = Task(
        description=f"""基于上面的学员画像，生成今天（第5天）的社群话术：
    1. 早安问候（适合发到班级群）
    2. 针对张阿姨的个性化鼓励（私聊发的）
    3. 今日打卡引导（怎么引导大家晒作业）
    4. 晚间总结（怎么引导大家说说今天的收获）

    要求：语气亲切，像跟邻居聊天，不要营销感。""",
        expected_output="4段可直接复制的话术，每段50字左右。",
        agent=speaker,
    )

    # ---------- 智能体3：作业点评员 ----------
    grader = Agent(
        role="作业点评员",
        goal="点评学员提交的作业，给出鼓励和具体改进建议",
        backstory="""你是书法老师，教了20年中老年学员。
    你点评作业先夸再提建议，语气鼓励为主，让学员有信心继续学。""",
        verbose=True,
        allow_delegation=False,
        llm=deepseek_llm,
    )
    task3 = Task(
        description=f"""点评张阿姨今天的作业：
    作业内容：临摹行书，张阿姨说“老师写得真好，我还是写不好”

    输出：
    1. 先夸（具体夸哪里）
    2. 再提一个小建议（不要多，就一个）
    3. 最后鼓励（让她有信心继续）""",
        expected_output="一段100字左右的作业点评，温暖、具体、有鼓励。",
        agent=grader,
    )

    # ---------- 智能体4：转化推荐员 ----------
    closer = Agent(
        role="转化推荐员",
        goal="根据学员7天数据，判断今天该不该推正价课，生成个性化转化话术",
        backstory="""你是中老年课程转化专家，最懂什么时候该推、什么时候该等。
    你推课不硬推，从学员的需求和兴趣出发，让学员觉得“这是为我好”。""",
        verbose=True,
        allow_delegation=False,
        llm=deepseek_llm,
    )
    task4 = Task(
        description=f"""基于上面所有分析，判断张阿姨今天（第5天）该不该推正价课：
    1. 该不该推？（该推/再等1天/还早）
    2. 为什么？
    3. 如果该推，生成一段私聊转化话术（从她的“想写好字”需求出发）

    学员第5天问过“正价课多少钱”。""",
        expected_output="转化判断+一段可直接发的私聊话术。",
        agent=closer,
    )

    # ---------- 组装Crew并执行 ----------
    crew = Crew(
        agents=[analyst, speaker, grader, closer],
        tasks=[task1, task2, task3, task4],
        process=Process.sequential,
        verbose=True,
    )
    final = crew.kickoff()

    outputs = {}
    for section, task in [
        ("学员画像", task1),
        ("社群话术", task2),
        ("作业点评", task3),
        ("转化建议", task4),
    ]:
        raw = getattr(getattr(task, "output", None), "raw", None)
        outputs[section] = str(raw).strip() if raw else ""
    if not outputs["转化建议"]:
        outputs["转化建议"] = str(final).strip()
    return outputs


if __name__ == "__main__":
    sections = run_conversion_assistant()
    print("\n" + "=" * 60)
    print("银发智脑 · 体验课转化助手 最终输出")
    print("=" * 60)
    for name, text in sections.items():
        print(f"\n【{name}】\n{text}")
