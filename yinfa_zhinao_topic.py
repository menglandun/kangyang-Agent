"""银发智脑 · 兴趣教育智慧中台 —— 第2批：爆款选题分析（2个智能体）。

- 独立运行：python yinfa_zhinao_topic.py（先在 .env 或环境变量里配置 DEEPSEEK_API_KEY）
- 应用集成：from yinfa_zhinao_topic import run_topic_analysis
"""
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from yinfa_zhinao_llm import build_deepseek_llm

load_dotenv()

# ---------- 模拟输入数据 ----------
BENCHMARK_DATA = """
参考账号：兴趣岛、十方、简知、小糖科技 近期爆款视频主题：
1. "60岁阿姨学书法，3个月变化太大了"（点赞10万+）
2. "退休后别跳广场舞了，来学这个"（点赞8万+）
3. "我妈学了智能手机，现在天天跟老外视频"（点赞15万+）
4. "60岁开始学国画，晚不晚？"（点赞6万+）
5. "退休老师教你写好名字，就3笔"（点赞12万+）
6. "我爸学了手机摄影，把我妈拍成了明星"（点赞20万+）
7. "60后妈妈的第一堂钢琴课"（点赞9万+）
8. "为什么建议60岁以上的人学点新东西？"（点赞7万+）
"""


def run_topic_analysis(benchmark_data=None):
    """运行2个智能体，返回各环节输出（爆款规律与选题清单/标题与钩子优化）。"""
    benchmark_data = benchmark_data or BENCHMARK_DATA

    deepseek_llm = build_deepseek_llm()

    # ---------- 智能体1：爆款选题分析员 ----------
    topic_analyst = Agent(
        role="短视频爆款选题分析员",
        goal="从同类账号的爆款视频中总结规律，输出本周可执行的选题清单",
        backstory="""你是中老年短视频领域的爆款操盘手，研究过几百个百万粉丝的中老年账号。
    你一眼就能看出什么选题容易火、为什么火、怎么改成自己的版本。""",
        verbose=True,
        allow_delegation=False,
        llm=deepseek_llm,
    )
    task1 = Task(
        description=f"""分析以下同类账号的爆款视频，输出：
    1. 爆款规律总结（3-5条，比如"真人前后对比""退休反差""子女视角"等）
    2. 本周10个可直接用的选题（每个选题含：标题、钩子开头、为什么容易火、适合拍什么内容）

    爆款参考：
    {benchmark_data}""",
        expected_output="一份爆款规律总结 + 10个本周可直接拍的选题清单。",
        agent=topic_analyst,
    )

    # ---------- 智能体2：选题文案优化员 ----------
    copywriter = Agent(
        role="短视频文案优化员",
        goal="把选题清单里的每个标题和钩子开头，改写成更抓人的版本",
        backstory="""你是短视频文案高手，最懂中老年用户刷到视频时的心理。
    你写的标题3秒就能让人停下来，钩子开头一句话就让人想看完。""",
        verbose=True,
        allow_delegation=False,
        llm=deepseek_llm,
    )
    task2 = Task(
        description="""基于上面的选题清单，把每个选题的标题和钩子开头优化一下：
    1. 标题要更抓人（用数字、反差、悬念）
    2. 钩子开头要3秒留人
    3. 语气像跟邻居聊天，不要太营销""",
        expected_output="优化后的10个选题，每个含最终标题+钩子开头。",
        agent=copywriter,
    )

    # ---------- 组装Crew并执行 ----------
    crew = Crew(
        agents=[topic_analyst, copywriter],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True,
    )
    final = crew.kickoff()

    outputs = {}
    for section, task in [
        ("爆款规律与选题清单", task1),
        ("标题与钩子优化版", task2),
    ]:
        raw = getattr(getattr(task, "output", None), "raw", None)
        outputs[section] = str(raw).strip() if raw else ""
    if not outputs["标题与钩子优化版"]:
        outputs["标题与钩子优化版"] = str(final).strip()
    return outputs


if __name__ == "__main__":
    sections = run_topic_analysis()
    print("\n" + "=" * 60)
    print("银发智脑 · 爆款选题分析 最终输出")
    print("=" * 60)
    for name, text in sections.items():
        print(f"\n【{name}】\n{text}")
