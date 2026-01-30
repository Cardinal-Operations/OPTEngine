### Step Deepseek-1
python3 s1_api_generate.py --solver_name gurobi --model_name gpt-5-2025-08-07 --data_path /data1/SIRL/onedayDiagnosis/test_data/NL4OPT.jsonl
python3 s1_api_generate.py --solver_name gurobi --model_name gpt-5 --data_path /data1/SIRL/onedayDiagnosis/test_data/MAMO_EasyLP_fixed.jsonl
python3 s1_api_generate.py --solver_name gurobi --model_name gpt-5 --data_path /data1/SIRL/onedayDiagnosis/test_data/DMAMO_ComplexLP_fixed.jsonl
python3 s1_api_generate.py --solver_name gurobi --model_name gpt-5 --data_path /data1/SIRL/onedayDiagnosis/test_data/IndustryOR_fixed.json
python3 s1_api_generate.py --solver_name gurobi --model_name gpt-5 --data_path /data1/SIRL/onedayDiagnosis/test_data/OptMATH_Bench_166.jsonl
python3 s1_api_generate.py --solver_name gurobi --model_name gpt-5 --data_path /data1/SIRL/onedayDiagnosis/test_data/OptiBench.jsonl
### The openAI-model


#python3 s2_deepseek_process.py --data_path ./api_return/output_s1/response_o3-2025-04-16_gurobi_MAMOEasyLPfixed.json
#python3 s2_deepseek_process.py --data_path ./api_return/output_s1/response_o3-2025-04-16_gurobi_MAMOComplexLPfixed.json
python3 s2_deepseek_process.py --data_path ./api_return/output_s1/response_gpt-5_gurobi_NL4OPT.json
python3 s2_deepseek_process.py --data_path ./api_return/output_s1/response_gpt-5_gurobi_EasyLPfixed.json
python3 s2_deepseek_process.py --data_path ./api_return/output_s1/response_gpt5_gurobi_MAMOComplexLPfixed.json
python3 s2_deepseek_process.py --data_path ./api_return/output_s1/response_gpt-5_gurobi_IndustryORfixed.json
python3 s2_deepseek_process.py --data_path ./api_return/output_s1/response_gpt-5_gurobi_OptMATHBench166.json
python3 s2_deepseek_process.py --data_path ./api_return/output_s1/response_gpt-5_gurobi_OptiBench.json

