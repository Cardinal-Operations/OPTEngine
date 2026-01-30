"""
The inference framework for the vllm
"""
import ast
import re,os
import json
import argparse
import numpy as np
from copy import deepcopy

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
from vllm import LLM, SamplingParams
from langchain_core.prompts import PromptTemplate

from utils import load_jsonl,check_result, check_pipeline_result
from benchmark_prompt_utils import benchmark_gurobi_prompts

def get_sampling_params(args):
    stop_tokens = ['<|im_end|>',"</s>","[/INST]"]    #For qwen models
    decoding_type = args.decoding_type
    # Base parameters used across all methods
    base_params = {
        "n": 1,
        "max_tokens": args.max_tokens,
        "stop": stop_tokens,
        "repetition_penalty": 1.05
    }
    if decoding_type == "top_p":
        return SamplingParams(temperature=0.5, top_p=0.95, **base_params)
    elif decoding_type == "greedy":
        return SamplingParams( temperature=0.0,top_p=1.0,top_k=1,   **base_params)
    elif decoding_type == "min_p":
        return SamplingParams(temperature=0.8,min_p=0.05,**base_params)
    elif decoding_type == "beamsearch":
        return SamplingParams(temperature=0.0,top_p=1.0, use_beam_search=True, best_of=5,  **base_params)
    else:
        raise ValueError(f"Unsupported decoding method: {decoding_type}")

def apply_prompt_template(item, prompt_type):
    sub_question = item['augmented']
    if prompt_type == "user":
        prompt = [{
        "role": "user",
        "content": zeroshot_prompt.format(Question=item['augmented']).strip() }]
    else:
        prompt = [
        {
        "role": "system",
        "content": zeroshot_prompt_system.format(Question=item['augmented']).strip() },
        {
        "role": "user",
        "content": zeroshot_prompt_user.format(Question=item['augmented']).strip() }]
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return text

def generate_with_model(model, prompt, sampling_params,batch_size):
    results = []
    for i in range(0, len(prompt), batch_size):
        batch = prompt[i:i + batch_size]
        response = model.generate(batch, sampling_params)
        batch_texts = [g.outputs[0].text for g in response]
        results.extend(batch_texts)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### data, model, prompt template, tool-type
    parser.add_argument('--data_path', default='/home/chenyitian/PolicyDistillation/data/newPipe/knapsack')#数据加载路径
    parser.add_argument("--data_name",   default = 'knapsack_rephrase.jsonl') ##, 'OptMATH_Bench_166.jsonl','MAMO_ComplexLP_fixed.jsonl']" ) #test_data
    parser.add_argument('--output_path', default='/home/chenyitian/PolicyDistillation/OREval/BaseModel')#数据加载路径
    parser.add_argument('--model_name', default='/DATA/disk1/chenyitian/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507')#
    parser.add_argument('--checkpoint', type=str, required=False, default=None, help='The path of the checkpoint after training')
    parser.add_argument('--prompt_type', type=str, required=False, default='user', choices=['user','system'], help='prompt type, options: []')
    parser.add_argument('--prompt_name', type=str, required=False, default='zeroshot_q2mc_v2', help='prompt name, options: []')
    parser.add_argument('--solver_name', type=str, required=False, default='gurobi',choices=['gurobi','copt','text'], help='solver name, options: []')

    parser.add_argument("--vllm_dype", type=str, required=False, default='bfloat16',choices=['bfloat16','float16'], help="Current precision type.")
    parser.add_argument('--tensor_parallel_size', type=int, required=False, default=4, help='tensor parallel size, options: []')
    parser.add_argument("--decoding_type", type=str, default="top_p", choices=["greedy", "top_p", "beamsearch", "min_p"],help="Decoding method")
    parser.add_argument("--batch_size", type=int, default=128, help="decoding batch-size")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Maximum number of tokens to generate")

    args = parser.parse_args()

    # Get the input keys from the arguments
    model_path = args.model_name            # The base model
    checkpoint = args.checkpoint  # The adapter or safetensor after training
    data_path = args.data_path              # The path of the testing dataset
    data_name = args.data_name              # The testing data, in json format
    data_name = [item.strip() for item in data_name.split(',')]

    prompt_name = args.prompt_name          # The prompt template
    prompt_type = args.prompt_type          # The prompt template
    solver_name = args.solver_name          # [gurobi, copt, text]

    sampling_params = get_sampling_params(args)
    batch_size = args.batch_size            # The batch_size for decoding

    # Load the model checkpoint and tokenizer, then initialize them for inference.
    # 如果checkpoint下有文件直接加载，否则把adapter转成pretrained再加载
    if args.checkpoint:
        print("transfer adapter to safetenor-format checkpoint")
        output_path = os.path.join(checkpoint,"pretrained")
        if not os.path.exists(output_path): hf_model()
        model = LLM(model=output_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.vllm_dype,
            trust_remote_code=True,
            gpu_memory_utilization=0.7)

    else:
        model = LLM(model=model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.vllm_dype,
            gpu_memory_utilization=0.7,
            trust_remote_code=True)
    print("Model initialized.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    ### The prompt templates
    if solver_name == "gurobi":
        zeroshot_prompt =  benchmark_gurobi_prompts[args.prompt_name]
        zeroshot_prompt = PromptTemplate.from_template(zeroshot_prompt)

    ### Load the dataset
    test_dataset = []
    for sub_name in data_name[:]:
        sub_datapath = os.path.join(args.data_path,sub_name)
        print("sub_datapath",sub_datapath)
        loaded_data = load_jsonl(sub_datapath)
        
        print("loaded_data", loaded_data)
        # generate responses
        prompt_list = []
        for item in loaded_data:
            prompt_list.append(apply_prompt_template(item,prompt_type))

        result_strs = generate_with_model(model, prompt_list, sampling_params,batch_size)
        snippet_package_abs = []
        for result_str,item in zip(result_strs,loaded_data):
            snippet_package_abs.append(check_pipeline_result(result_str,item, solver_name))
    
        result_key = {0: 'wrong',
            1: 'correct',
            3: 'excution error',
            2: 'code formulation failed',
            4: 'other error'
            }
        sub_result_count = np.bincount(snippet_package_abs)
        sub_result_accuracy = round(sub_result_count[1]/sum(sub_result_count),4)

        ## Output the response for diagnostics 
        sub_data_prefix = sub_name.split('.')[0]
        sub_model_name = args.model_name.split('/')[-1]
    
        output_path = args.output_path
        os.makedirs(output_path, exist_ok=True)
        ## The statistics results
        accuracy_eval_path = os.path.join(output_path,f'{sub_model_name}_{sub_data_prefix}_pass@1.txt')
        with open(accuracy_eval_path,'a',encoding='utf-8') as fp:
            fp.write(f'{sub_data_prefix}--pass@1 accuracy: {sub_result_accuracy}; count stats: {sub_result_count}\n\n')

        ## The responses
        output_response_path = os.path.join(output_path,f'{sub_model_name}_{sub_data_prefix}_{args.decoding_type}_response.jsonl')
        with open(output_response_path,'w',encoding='utf-8') as f:
            for result_str,item, error_type in zip(result_strs,loaded_data, snippet_package_abs):
                output_item = deepcopy(item)
                output_item['response'] = result_str
                output_item['error_type'] = error_type
                f.write(json.dumps(output_item) + '\n')
