import os,re
import json
import subprocess
import numpy as np

def load_jsonl(filepath):
    """Loads a JSONL (JSON Lines) file and returns a list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()}")
                    print(f"Error details: {e}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []
    return data

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

def check_result(result_str, item,solver_name):
    sub_answer = item['en_answer']
    if sub_answer == "No Best Solution" or "-9999" in str(sub_answer):
        sub_answer = None
    else:
        sub_answer = float(item['en_answer'])
    code_snippet = extract_code_block(result_str,solver_name)
    if code_snippet is None:
        return 2
    try:
        result = subprocess.run(['python3', '-c', code_snippet], capture_output=True, text=True, timeout=100)
    except subprocess.TimeoutExpired as e:
        if sub_answer is None:
            return 1
        else:
            return 0
    if result.returncode !=0 :
        return 3
    solver_result = extract_obj(result.stdout)
    if sub_answer and solver_result:
        return int(np.abs(solver_result-sub_answer)<0.01)
    elif sub_answer == solver_result:
        return 1
    elif 'nfeasible' in result.stdout:
        if sub_answer is None:
            return 1
        else:
            return 0
    else:
        return 4

def check_pipeline_result(result_str, item,solver_name):
    sub_answer = item['true_answer']
    if sub_answer == "No Best Solution" or "-9999" in str(sub_answer):
        sub_answer = None
    else:
        sub_answer = float(item['true_answer'])
    code_snippet = extract_code_block(result_str,solver_name)
    if code_snippet is None:
        return 2
    try:
        result = subprocess.run(['python3', '-c', code_snippet], capture_output=True, text=True, timeout=100)
    except subprocess.TimeoutExpired as e:
        if sub_answer is None:
            return 1
        else:
            return 0
    if result.returncode !=0 :
        return 3
    solver_result = extract_obj(result.stdout)
    if sub_answer and solver_result:
        return int(np.abs(solver_result-sub_answer)<0.01)
    elif sub_answer == solver_result:
        return 1
    elif 'nfeasible' in result.stdout:
        if sub_answer is None:
            return 1
        else:
            return 0
    else:
        return 4
