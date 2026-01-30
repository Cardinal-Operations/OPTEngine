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
    """
    0: do not find tags
    1: could not extract numeric value
    2: not correct
    3: correct
    """
    item =  deepcopy(item2)
    sub_question = item['en_question']
    sub_answer = item['en_answer']
    if sub_answer == "No Best Solution":
        sub_answer = None
    else:
        sub_answer = float(item['en_answer'])

    result_str = item['response']
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, result_str, re.IGNORECASE | re.DOTALL)
    if match:
        # 3. Clean and Extract Content:
        # Get the content from the capture group and strip leading/trailing whitespace.
        content = match.group(1).strip()

        # We search the extracted 'content' only.
        number_pattern = r'[-+]?\s*(\d*\.?\d+)'
        number_match = re.search(number_pattern, content)

        if number_match:
            try:
                # Convert the captured numerical string to a floating-point number.
                answer= float(number_match.group(0).strip())
                if(np.abs(answer-sub_answer)<1e-6):
                    return 3
                else:
                    return 2
            except ValueError:
                # Should not happen if the regex is correct, but handles unexpected formats.
                print(f"Warning: Extracted content '{number_match.group(0)}' is not a valid number.")
                return 1
        else:
            print(f"Warning: Found tags but could not extract a number from '{content}'.")
            return 2
    else:
        # Tags not found in the output.
        return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--data_path', type=str, required=False, default='/data1/SIRL/verbalBenchmark/api_return/output_s1/response_deepseek-chat_gurobi_IndustryOR_fixedV2.json', help='path of the test data, options: []')
        
    args = parser.parse_args()
    filepath = args.data_path
    filepath = "/data1/SIRL/verbalBenchmark/api_return/output_s1/response_deepseek-chat_gurobi_IndustryOR_fixedV2.json"

    filename_str = filepath.split('/')[-1]
    filename_list = filename_str.split('_')
    model_name,solver_name, data_name = filename_list[1], filename_list[2], filename_list[3]


    loaded_data = load_jsonl(filepath)
    if loaded_data:
        pass
        print(f"Successfully loaded {len(loaded_data)} items from {filepath}")
    test_data =  loaded_data[:]

    max_workers=32

    error_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(mp_worker, task) for task in test_data]
        for sub_result in tqdm(as_completed(futures), total=len(test_data), desc="Processing"):
            error_results.append(sub_result.result())
 
    result_key = {0: 'not found tag',
            1: 'extract numeric value failed',
            2: 'wrong answer',
            3: 'correct'
            }
    result = np.bincount(error_results)

    """
    save the results
    """
    save_path = f"api_return/output_s2/stats_{model_name}_{solver_name}_{data_name}.json"
    if not os.path.exists(f"api_return/output_s2"):
        os.mkdir(f"api_return/output_s2")

    with open(save_path,'a',encoding = 'utf-8') as f:
        f.write(f'pass@1 accuracy for dataset {data_name}: {result[3]}/{sum(result)} = {result[3] / sum(result)}, error types: {result}\n')

