### Step Deepseek-1
python3 s1_deepseek_generate.py --solver_name gurobi --model_name deepseek-chat --data_path ./test_data/IndustryOR_fixedV2.json
python3 s1_deepseek_generate.py --solver_name gurobi --model_name deepseek-chat --data_path ./test_data/MAMO_ComplexLP_fixed.jsonl
python3 s1_deepseek_generate.py --solver_name gurobi --model_name deepseek-chat --data_path ./test_data/OptMATH_Bench_166.jsonl

### Stats the results
python3 s2_deepseek_process.py --data_path ./api_return/output_s1/response_deepseek-chat_gurobi_OptMATH_Bench_166.json
python3 s2_deepseek_process.py --data_path ./api_return/output_s1/response_deepseek-chat_gurobi_IndustryOR_fixedV2.json
python3 s2_deepseek_process.py --data_path ./api_return/output_s1/response_deepseek-chat_gurobi_MAMO_ComplexLP_fixed.json


