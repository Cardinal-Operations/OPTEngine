import os
import time, re
import argparse
import json
import itertools
import tiktoken
import subprocess
from copy import deepcopy
from openai import OpenAI
from langchain.prompts import PromptTemplate
import multiprocessing
from utils import load_jsonl, generate_with_api ,execute_code_output
from content_utils import extract_code_block,extract_obj
from benchmark_prompt_utils import benchmark_gurobi_prompts
import numpy as np
from concurrent.futures import ThreadPoolExecutor,as_completed
from tqdm import tqdm
import threading

## for code and mask tasks
completion_kwargs = {
    "temperature": 0.5,
    "top_p": 0.95,
    "n": 1,
}

lock = threading.Lock()

def mp_worker(item_product):
    item_product2 = deepcopy(item_product)
    model_name, item = item_product2 
    sub_question = item['en_question' ]
    """
    if sub_answer == "No Best Solution":
        sub_answer = None
    else:
        sub_answer = float(item['en_answer'])
    """
    question_str = zeroshot_prompt.format(Question=sub_question).strip()
    result_str = generate_with_api(client,question_str, model_name, completion_kwargs)
    
    item['response'] = result_str
    item['model_name'] = model_name
    return item


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--model_name', type=str, required=True, default="gpt-5", help='model name, options: []')
    parser.add_argument('--prompt_name', type=str, required=False, default='zeroshot_q2mc_en', help='prompt name, options: []')
    parser.add_argument('--solver_name', type=str, required=False, default='gurobi',
                        help='solver name, options: []')
    parser.add_argument('--data_path', type=str, required=False, default='/data1/SIRL/Diagnosis/test_data/NL4OPT.jsonl',
                        help='path of the test data, options: []')
    
    args = parser.parse_args() 
    model_name = args.model_name
    solver_name = args.solver_name
    prompt_name = args.prompt_name
    filepath = args.data_path
    data_name =  filepath.split("/")[-1].split(".")[0]

    method = "top_p"
    api_key = "sk-mqhNkt4RQZr9a8sX03078a0875E543E5962b23055d5aA064"
    base_url = "https://api.ai-gaochao.cn/v1"

    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    client = OpenAI(api_key=api_key, base_url=base_url)

    print("model name:\t", model_name)
    print("solver name:\t", solver_name)
    print("prompt name:\t", prompt_name)
    print("data path:\t", filepath)
    
    ### The testing dataset
    loaded_data = load_jsonl(filepath)
    if loaded_data:
        pass
        print(f"Successfully loaded {len(loaded_data)} items from {filepath}")
    test_data =  loaded_data[:]

    zeroshot_prompt =  benchmark_gurobi_prompts[args.prompt_name]
    zeroshot_prompt = PromptTemplate.from_template(zeroshot_prompt)

    max_workers=32
    
    model_list = [model_name]
    comb_list = list(itertools.product(model_list, test_data))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(mp_worker, task) for task in comb_list]
        for sub_result in tqdm(as_completed(futures), total=len(test_data), desc="Processing"):
            results.append(sub_result.result())

    save_path = f"api_return/output_s1/response_{model_name}_{solver_name}_{data_name}.json"
    if not os.path.exists(f"api_return/output_s1"):
        os.mkdir(f"api_return/output_s1")

    with open(save_path,'w',encoding='utf-8') as f:
        for item in results:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
