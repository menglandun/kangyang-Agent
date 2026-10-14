# pip install langgraph langchain-openai langgraph-checkpoint-sqlite
import ast
import math
import os
import re
from pathlib import Path
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command

DB_PATH = Path(__file__).resolve().parent / "checkpoints.db"

# ============ 1. 配置 DeepSeek ============
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY，例如：export DEEPSEEK_API_KEY=sk-xxxx")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0,
)

# ============ 2. 定义工具 ============
@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""
    fake_db = {"北京": "晴 12°C", "上海": "多云 18°C", "伦敦": "阴 15°C"}
    return fake_db.get(city, f"{city} 暂无天气数据")

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件（需要人工审批后才真正执行）。"""
    return f"邮件已发送至 {to}，主题：{subject}"

# ============ 计算器安全求值（AST 白名单 + 硬上限）============
CALC_MAX_EXPRESSION_LENGTH = 200
CALC_MAX_INT_DIGITS = 30
CALC_MAX_EXPONENT = 1000
CALC_MAX_RESULT_DIGITS = 5000
CALC_MAX_FACTORIAL_N = 1000


class _CalcLimitError(Exception):
    """计算超出安全上限。"""


def _calc_int_digits(value: int) -> int:
    return len(str(abs(value)))


def _calc_check_result(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise _CalcLimitError("结果超出范围")
    if isinstance(value, int) and _calc_int_digits(value) > CALC_MAX_RESULT_DIGITS:
        raise _CalcLimitError("结果规模过大")


def _calc_eval_node(node):
    if isinstance(node, ast.Expression):
        return _calc_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        if isinstance(node.value, int) and _calc_int_digits(node.value) > CALC_MAX_INT_DIGITS:
            raise _CalcLimitError("数字位数过多")
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _calc_eval_node(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.Call):
        # 仅允许后置阶乘改写成受控的 fact(n) 调用
        if (isinstance(node.func, ast.Name) and node.func.id == "fact"
                and len(node.args) == 1 and not node.keywords):
            n = _calc_eval_node(node.args[0])
            if isinstance(n, float):
                n = int(n)
            if not isinstance(n, int) or n < 0 or n > CALC_MAX_FACTORIAL_N:
                raise _CalcLimitError("阶乘数值超出范围（n 需在 0~1000）")
            result = math.factorial(n)
            _calc_check_result(result)
            return result
        raise _CalcLimitError("表达式包含不支持的内容")
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left = _calc_eval_node(node.left)
        right = _calc_eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if not isinstance(right, (int, float)) or abs(right) > CALC_MAX_EXPONENT:
                raise _CalcLimitError("幂指数过大")
            if left != 0 and abs(left) > 1:
                if abs(right) * math.log10(abs(left)) > CALC_MAX_RESULT_DIGITS:
                    raise _CalcLimitError("结果规模过大")
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        else:
            result = left ** right
        _calc_check_result(result)
        return result
    raise _CalcLimitError("表达式包含不支持的内容")


def _safe_calc(expression: str) -> str:
    """支持 + - * / ** 与后置阶乘 !；超限时返回中文错误，不执行用户代码。"""
    allowed_chars = set("0123456789+-*/().! \t")
    if len(expression) > CALC_MAX_EXPRESSION_LENGTH:
        return "表达式过长，已拒绝计算"
    if any(c not in allowed_chars for c in expression):
        return "表达式包含非法字符"
    for num in re.findall(r"\d+(?:\.\d+)?", expression):
        if sum(c.isdigit() for c in num) > CALC_MAX_INT_DIGITS:
            return "表达式中数字位数过多"
    expr = re.sub(r"(\d+(?:\.\d+)?)!", r"fact(\1)", expression)
    try:
        tree = ast.parse(expr, mode="eval")
        result = _calc_eval_node(tree)
        _calc_check_result(result)
    except Exception as e:
        return f"计算失败：{e}"
    return f"{expression} = {result}"


@tool
def calculator(expression: str) -> str:
    """计算数学表达式，例如 '(5+7)*3' 或 '10!'。"""
    return _safe_calc(expression)

@tool
def search_info(query: str) -> str:
    """搜索知识，返回事实性答案。"""
    db = {"法国首都": "巴黎", "日本首都": "东京", "地球人口": "约80亿"}
    for k, v in db.items():
        if k in query:
            return f"{k}：{v}"
    return f"未找到关于 '{query}' 的信息"

@tool
def query_employee(department: str) -> str:
    """查询某部门的员工名单。"""
    fake_db = {"技术部": ["张伟", "李娜", "王强"], "市场部": ["陈静", "赵磊"], "财务部": ["周敏"]}
    employees = fake_db.get(department)
    return f"{department} 员工：{', '.join(employees)}" if employees else f"可选：技术部、市场部、财务部"

tools = [get_weather, send_email, calculator, search_info, query_employee]
llm_with_tools = llm.bind_tools(tools)

# ============ 3. State ============
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ============ 4. 节点 ============
def call_llm(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def human_approval(state: AgentState):
    """中断点：执行 send_email 前暂停；interrupt() 会把状态存进 checkpointer。"""
    tool_calls = state["messages"][-1].tool_calls
    email_call = next(tc for tc in tool_calls if tc["name"] == "send_email")
    approved = interrupt({
        "action": "需要审批发邮件",
        "tool": email_call["name"],
        "args": email_call["args"],
        "tool_calls": [{"name": tc["name"], "args": tc["args"]} for tc in tool_calls],
    })
    if approved:
        # 人工批准：整批调用交给 tools 节点执行
        return {}
    # 人工拒绝：为本批每个 tool_call 逐条补 ToolMessage，保证消息契约完整
    return {
        "messages": [
            ToolMessage(
                content=f"用户拒绝了本次操作（{tc['name']}），未执行。",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

tool_node = ToolNode(tools)

def route_after_llm(state: AgentState):
    last_msg = state["messages"][-1]
    if not last_msg.tool_calls:
        return END
    return "human_approval" if any(tc["name"] == "send_email" for tc in last_msg.tool_calls) else "tools"

def route_after_approval(state: AgentState):
    """审批通过 -> 执行工具；被拒绝 -> 带着拒绝结果回到 LLM。"""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, ToolMessage):
        return "llm"
    return "tools"

# ============ 5. 构建图 ============
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("human_approval", human_approval)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", route_after_llm, {
    "human_approval": "human_approval", "tools": "tools", END: END,
})
graph.add_conditional_edges("human_approval", route_after_approval, {
    "tools": "tools", "llm": "llm",
})
graph.add_edge("tools", "llm")

# ============ 6. 多用户交互 ============
if __name__ == "__main__":
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        print("=== 多用户智能体（输入 exit 退出）===\n")
        user_id = input("请输入你的用户ID（如 alice / bob）：").strip()
        config = {"configurable": {"thread_id": user_id}}
        print(f"已登录：{user_id}\n")

        # 如果这个用户上次有没审批完的邮件，先在这里恢复
        existing = app.get_state(config)
        if existing and existing.next and existing.tasks and existing.tasks[0].interrupts:
            print("检测到未完成的审批：", existing.tasks[0].interrupts[0].value)
            decision = input("批准？(y/n): ").strip().lower() == "y"
            r0 = app.invoke(Command(resume=decision), config)
            print("AI：", r0["messages"][-1].content)
            print()

        while True:
            user_input = input("你：").strip()
            if user_input.lower() in ("exit", "quit"):
                print("再见！")
                break
            if not user_input:
                continue

            result = app.invoke({"messages": [("user", user_input)]}, config)

            # 处理中断
            snapshot = app.get_state(config)
            if snapshot.tasks and snapshot.tasks[0].interrupts:
                print("审批请求：", snapshot.tasks[0].interrupts[0].value)
                decision = input("批准？(y/n): ").strip().lower() == "y"
                result = app.invoke(Command(resume=decision), config)

            print("AI：", result["messages"][-1].content)
            print()
