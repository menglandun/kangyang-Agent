"""银发智脑 · 共用 LLM 配置。

各智能体模块统一从这里获取 DeepSeek LLM：换模型、调温度只改这一处。
"""
import os

from dotenv import load_dotenv
from crewai import LLM

load_dotenv()


def build_deepseek_llm():
    """构建 DeepSeek LLM（密钥从环境变量或 .env 读取）。"""
    return LLM(
        model="deepseek/deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 密钥放在环境变量或 .env 中，不要写死在代码里
        temperature=0.7,
    )
