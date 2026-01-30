import os,re
import json
import argparse
from copy import deepcopy
from openai import OpenAI

import multiprocessing
from utils import execute_code_output, extract_code_block,extract_obj_and_sol,insert_print
# from content_utils import extract_obj
from benchmark_prompt_utils import benchmark_gurobi_prompts
import numpy as np
from concurrent.futures import ThreadPoolExecutor,as_completed
from tqdm import tqdm
import threading
import subprocess
from utils import  load_jsonl
from content_utils import extract_code_block,extract_obj

def mp_worker(item2):
    item =  deepcopy(item2)
    sub_question = item['en_question']
    sub_answer = item['en_answer']
    if sub_answer == "No Best Solution":
        sub_answer = None
    else:
        sub_answer = float(item['en_answer'])
    #item = loaded_data[99]
    result_str = item['response']
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--data_path', type=str, required=False, default='/data1/SIRL/Diagnosis/api_return/output_s1/response_deepseek-chat_gurobi_MAMOComplexLPModified.json', help='path of the test data, options: []')
        
    args = parser.parse_args()
    filepath = args.data_path
    filename = filepath.split('/')[-1].split('.')[0]
    filename_list = filename.split('_')
    model_name,solver_name = filename_list[1], filename_list[2]
    data_name = ''.join(filename_list[3:])
    loaded_data = load_jsonl(filepath)
    if loaded_data:
        pass
        print(f"Successfully loaded {len(loaded_data)} items from {filepath}")
    test_data =  loaded_data[:]

    max_workers=64

    error_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(mp_worker, task) for task in test_data]
        for sub_result in tqdm(as_completed(futures), total=len(test_data), desc="Processing"):
            error_results.append(sub_result.result())
 
    result_key = {0: 'wrong',
            1: 'correct',
            3: 'excution error',
            2: 'code formulation failed',
            4: 'other error'
            }
    result = np.bincount(error_results)

    """
    save the results
    """
    save_path = f"api_return/output_s2/stats_{model_name}_{solver_name}_{data_name}.json"
    if not os.path.exists(f"api_return/output_s2"):
        os.mkdir(f"api_return/output_s2")

    with open(save_path,'a',encoding = 'utf-8') as f:
        f.write(f'pass@1 accuracy for dataset {data_name}: {result[1]}/{sum(result)} = {result[1] / sum(result)}, error types: {result}\n')

