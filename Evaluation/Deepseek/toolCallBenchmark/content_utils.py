import re
import subprocess
# ---------------------------------------
# 提取代码块的函数
# ---------------------------------------

def insert_print(code:str, solver_name:str) -> str:

    # 动态匹配模型名字
    model_pattern = r'^(\s*)(\w+)\.(optimize|solve)\(\)'
    model_match = re.search(model_pattern, code, re.M)
    if model_match:
        indent = model_match.group(1)  # 获取缩进
        model_name = model_match.group(2)  # 获取模型名字
        optimize_call = model_match.group(3)  # 获取优化调用方法

        # 根据求解器名称设置优化调用方法
        if solver_name == "gurobi":
            pattern = r'^(\s*)(' + model_name + r'\.optimize\(\))'
        elif solver_name == "copt":
            pattern = r'^(\s*)(' + model_name + r'\.solve\(\))'

        # 使用正则表达式替换，并保持相同的缩进
        code = re.sub(pattern, rf'\1\2\n{indent}print(f"Just print the best solution: {{{model_name}.ObjVal}}")', code, flags=re.M)
        return code
    return code

def extract_code_block(llm_output: str,solver_name) -> str:
    """
    使用正则提取三引号 ```python ...``` 之间的代码（DOTALL 模式）。
    若未匹配到则返回空字符串。
    """
    pattern = r'<python>(.*?)</python>'
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if '```' in code: #可能python内部额外加了代码块
            pattern = r'```python(.*?)```'
            match = re.search(pattern, code, re.DOTALL)
            if match:
                code = match.group(1).strip()
        code = insert_print(code, solver_name)
        return code
    # 可能没有pyhon符号
    pattern = r'```python(.*?)```'
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        code = insert_print(code,solver_name)
        return code
    return None



def extract_obj(str_log):
    """Extract objective value from log string"""
    if 'Just print the best solution:' in str_log:
        item = next(i for i in str_log.split('\n') if 'Just print the best solution:' in i)
        result = re.findall(r'-?\d+\.?\d*', item)
        return float(result[0]) if result else None
    return None
