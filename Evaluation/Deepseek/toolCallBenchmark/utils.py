import os
import time
import json
"""
The load data
"""
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

"""
The output
"""

def write_string_to_python_file(filepath, string_content, overwrite=False):
    """Writes a string to a Python (.py) file.

    Args:
        filepath: The path to the output .py file.
        string_content: The string to be written to the file.
        overwrite: If True, overwrites the file if it exists. If False, appends to the file. Defaults to False.

    Returns:
        True if the write operation was successful, False otherwise.  Also returns False if the file exists and overwrite is False.
        Prints informative messages about success or failure.
    """

    try:
        mode = 'w' if overwrite else 'a'  # 'w' for write (overwrite), 'a' for append
        if not overwrite and os.path.exists(filepath):
            print(f"File '{filepath}' already exists. Appending to the file.") # Informative message if appending
        with open(filepath, mode, encoding='utf-8') as f: # Handle encoding
            f.write(string_content)
        print(f"Successfully wrote/appended to '{filepath}'.")
        return True
    except Exception as e:
        print(f"An error occurred while writing to '{filepath}': {e}")
        return False



def save_to_markdown(text,  filepath, filename):
    """Saves a string to a Markdown (.md) file.

    Args:
        text: The input string.
        filepath: The full path to the output Markdown file (including the .md extension).
    """
    filepath = os.path.join(filepath, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as md_file:
            md_file.write(text)

        print(f"String saved to {filepath}")

    except Exception as e:
        print(f"Error creating Markdown file: {e}")


# ---------------------------------------
# OpenAI API 调用
# ---------------------------------------
def generate_with_deepseek_api(client, prompt: str, modelname):
    assert modelname in ['deepseek-chat', 'deepseek-reasoner']
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        messages=messages,
        model=modelname,
    )
    result_text = str(response.choices[0].message.content)
    return result_text

# ---------------------------------------
# OpenAI API 调用
# ---------------------------------------
def generate_with_api(client, prompt: str, modelname: str, completion_kwargs:dict):
    messages = [{"role": "user", "content": prompt}]
    if(modelname in ['gpt-4o-latest','gpt-5']):
        response = client.chat.completions.create(
            messages=messages,
            model=modelname,
        )
    result_text = str(response.choices[0].message.content)
    return result_text

def generate_with_reasoning_api(client, prompt: str, modelname):
    messages = [{"role": "user", "content": prompt}]
    if(modelname in ['deepseek-reasoner','o3','o3-2025-04-16']):
        response = client.chat.completions.create(
            messages=messages,
            model=modelname,
        )
    result_text = str(response.choices[0].message.content)
    return result_text


# ---------------------------------------
# doubao
# ---------------------------------------
def generate_with_doubao_api(client, prompt: str, modelname: str,completion_kwargs:dict):
    assert modelname in ["doubao"]
    messages = [{"role": "user", "content": prompt}]
    if(modelname=='doubao'):
        modelname = 'doubao-1.5-pro-32k-250115'
    response = client.chat.completions.create(
            messages=messages,
            model=modelname,
            **completion_kwargs
        )
    result_text = str(response.choices[0].message.content)
    return result_text

import subprocess
def execute_code_output(code):
    try:
        result = subprocess.run(['python3', '-c', code], capture_output=True, text=True, timeout=100)  # 设置超时时间为10秒
        output = result.stdout
        error = result.stderr
        return output, error
    except subprocess.TimeoutExpired:
        return None, "Execution timed out"
    except Exception as e:
        return None, str(e)

import re
def insert_print(code: str, solver_name: str) -> str:
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
            status_check = (
                f"{indent}if {model_name}.status == GRB.OPTIMAL:\n"
                f"{indent}    print(f'Just print the best obj: {{{model_name}.ObjVal}}')\n"
                f"{indent}    print('Just print the best sol:[', end = '')\n"
                f"{indent}    for var in {model_name}.getVars():\n"
                f"{indent}        print(f'{{var.X}}', end = ',')\n"
                f"{indent}    print(']')\n"
                f"{indent}else:\n"
                f"{indent}    print('No optimal solution found, status:', {model_name}.status)"
            )
        # 使用正则表达式替换，并保持相同的缩进
        code = re.sub(pattern, rf'\1\2\n{status_check}', code, flags=re.M)
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

def extract_obj_and_sol(str_log):
    """Extract objective value from log string"""
    if not str_log:
        return None, None
    obj = None
    sol = [None]
    if 'Just print the best obj:' in str_log:
        item = next(i for i in str_log.split('\n') if 'Just print the best obj:' in i)
        result = re.findall(r'-?\d+\.?\d*', item)
        if result:
            obj = float(result[0])
        else:
            obj = None
        #return float(result[0]) if result else None
    if 'Just print the best sol:' in str_log:
        sol_match = re.search(r'Just print the best sol:\s*\[([-\d.,\s]*)\]', str_log)
        best_sol = [float(x) for x in sol_match.group(1).split(',') if x.strip()] if sol_match else None
        if best_sol:
            best_sol.sort()
            sol = best_sol
        else:
            sol = [None]
    return (obj,sol)
