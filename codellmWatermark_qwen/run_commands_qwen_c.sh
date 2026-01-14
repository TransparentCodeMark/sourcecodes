#!/bin/bash

# 初始化conda
eval "$(conda shell.bash hook)"

# 激活conda环境
conda activate LLMBackdoorAttack_Finetune_CodeSummarization

# 切换到正确的目录
cd /home/qyb/SynologyDrive/project/codellmWatermark/

# 运行Python脚本
python Qwen2.5_coder-7B-Instruct-Lora_Finetune_C.py
python Qwen2.5_coder-7B-Instruct-Lora_Inference_C.py