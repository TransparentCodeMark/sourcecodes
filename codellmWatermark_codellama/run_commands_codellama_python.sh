#!/bin/bash

# 初始化conda
eval "$(conda shell.bash hook)"

# 激活conda环境
conda activate LLMBackdoorAttack_Finetune_CodeSummarization

# 切换到正确的目录
cd /home/qyb/SynologyDrive/project/codellmWatermark_codellama/

# 运行Python脚本
python codellama-Instruct-Lora_Finetune_Python.py --ratio 0.1
python codellama-Instruct-Lora_Inference_Python.py --ratio 0.1