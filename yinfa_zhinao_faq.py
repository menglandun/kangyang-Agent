"""银发智脑 · FAQ自动应答（独立模块）。

- 独立运行：python yinfa_zhinao_faq.py（先在 .env 或环境变量里配置 DEEPSEEK_API_KEY）
- 应用集成：from yinfa_zhinao_faq import run_faq_answers
"""
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from yinfa_zhinao_llm import build_deepseek_llm

load_dotenv()

# ---------- 模拟输入数据 ----------
DEFAULT_QUESTIONS = """1. 怎么登录学习APP？
2. 作业怎么提交？拍照还是打字？
3. 正价课怎么退款？
4. 上课时间可以自己选吗？
5. 学不会怎么办？有老师一对一教吗？
"""


def run_faq_answers(questions=None):
    """运行答疑客服，返回分节结果。"""
    questions = questions or DEFAULT_QUESTIONS

    deepseek_llm = build_deepseek_llm()

    responder = Agent(
        role="学员答疑客服",
        goal="用亲切易懂的话回答学员的常见问题",
        backstory="""你是中老年教育平台的客服，最擅长跟60后阿姨大叔沟通。
    你说话慢、清楚、有耐心，不用专业术语，像邻居一样帮人解决问题。""",
        verbose=True,
        allow_delegation=False,
        llm=deepseek_llm,
    )
    task = Task(
        description=f"""请把以下问题各写一段可以直接发给学员的回答：
    {questions}

    要求：
    1. 语气亲切，像跟邻居聊天
    2. 步骤清楚，不要跳步
    3. 每个回答控制在100字以内""",
        expected_output="每个问题配一段可直接发的回答。",
        agent=responder,
    )

    crew = Crew(agents=[responder], tasks=[task], process=Process.sequential, verbose=True)
    final = crew.kickoff()

    raw = getattr(getattr(task, "output", None), "raw", None)
    text = str(raw).strip() if raw else str(final).strip()
    return {"常见问题解答": text}


if __name__ == "__main__":
    sections = run_faq_answers()
    print("\n" + "=" * 60)
    print("银发智脑 · FAQ自动应答 最终输出")
    print("=" * 60)
    for name, text in sections.items():
        print(f"\n【{name}】\n{text}")
