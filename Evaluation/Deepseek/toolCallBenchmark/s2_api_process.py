import os
import time, re
import argparse
import json
import tiktoken
import subprocess
from openai import OpenAI
from langchain.prompts import PromptTemplate
import multiprocessing
from utils import load_jsonl, generate_with_reasoning_api
from content_utils import extract_code_block,extract_obj
from benchmark_prompt_utils import benchmark_gurobi_prompts
import numpy as np

def mp_worker(item):

    sub_question = item['en_question' ]
    sub_answer = item['en_answer']
    if sub_answer == "No Best Solution":
        sub_answer = None
    else:
        sub_answer = float(item['en_answer'])
    question_str = zeroshot_prompt.format(Question=sub_question).strip()
    result_str = generate_with_reasoning_api(client,question_str, model_name)
   
    code_snippet = extract_code_block(result_str,solver_name)
 
    if code_snippet is None:
        return 2
    try:
        result = subprocess.run(['python3', '-c', code_snippet], capture_output=True, text=True, timeout=200)
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
    parser.add_argument('--model_name', type=str, required=False, default="o3-2025-04-16", help='model name, options: []')
    parser.add_argument('--prompt_name', type=str, required=False, default='zeroshot_q2mc_en', help='prompt name, options: []')
    parser.add_argument('--solver_name', type=str, required=False, default='gurobi',
                        help='solver name, options: []')
    parser.add_argument('--data_path', type=str, required=False, default='/data3/nvme0n1/LPLLM/Rebuttal/TestDeepSeekR1/test_data',
                        help='path of the test data, options: []')
    
    args = parser.parse_args() 
    model_name = args.model_name
    solver_name = args.solver_name
    prompt_name = args.prompt_name
    filepath = args.data_path
    filename = ['NL4OPT.jsonl', 'MAMO_EasyLP.json', 'MAMO_ComplexLP.json', 'IndustryOR_fixed.json', 'OptMATH_Bench_193.jsonl', 'OptiBench.jsonl']
    filename = ['OptiBench.jsonl']
    method = "top_p"
    api_key = "sk-vVCsbzRAqnPPYvBc15C6AdB56aCd4fA980F5C6982eA67c8e"
    base_url = "https://api.ai-gaochao.cn/v1"

    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    client = OpenAI(api_key=api_key, base_url=base_url)


    print("model name:\t", model_name)
    print("solver name:\t", solver_name)
    print("prompt name:\t", prompt_name)
    for file in filename:
        dataset = os.path.join(filepath,file)
        print("test dataset:\t", dataset)

    ### The testing dataset
        loaded_data = load_jsonl(dataset)
        if loaded_data:
            pass
        #print(f"Successfully loaded {len(loaded_data)} items from {filepath}")
        test_data =  loaded_data[:]
    ### The zero-shot prompt testing
    #if args.solver_name == 'gurobi':
        zeroshot_prompt =  benchmark_gurobi_prompts[args.prompt_name]
    #elif args.solver_name == 'copt':
    #    zeroshot_prompt =  benchmark_copt_prompts[args.prompt_name]
        zeroshot_prompt = PromptTemplate.from_template(zeroshot_prompt)
        p = multiprocessing.Pool(24)
    #tasks = [(data_, model_name, solver_name) for data_ in data]
        snippet_package =p.map(mp_worker, test_data)
        p.close()
        result_key = {0: 'wrong',
            1: 'correct',
            3: 'excution error',
            2: 'code formulation failed',
            4: 'other error'
            }
        result = np.bincount(snippet_package)
        with open('O1pass@1_accuracy2.txt','a',encoding = 'utf-8') as f:
            f.write(f'pass@1 accuracy for dataset {dataset}: {result[1]}/{sum(result)} = {result[1] / sum(result)}\n')
        print(dataset, snippet_package)
        #print(result)
    #print(model_name, solver_name, prompt_name, filepath,result)
