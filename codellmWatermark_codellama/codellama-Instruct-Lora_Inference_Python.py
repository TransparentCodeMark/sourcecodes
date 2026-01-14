import pandas as pd
import torch
from datasets import Dataset
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from peft import PeftModel  # 将JSON文件转换为CSV文件
from rouge import Rouge
from transformers import AutoModelForCausalLM , AutoTokenizer


def find_latest_checkpoint ( base_path ) :
    # 确保base_path存在
    if not os.path.exists ( base_path ) :
        return None
    # 获取所有的checkpoint文件夹
    checkpoints = [ d for d in os.listdir ( base_path ) if d.startswith ( 'checkpoint-' ) ]
    if not checkpoints :
        return None
    # 提取数字并找到最大值
    max_num = max ( int ( re.findall ( r'\d+' , cp ) [ 0 ] ) for cp in checkpoints )

    # 构建完整路径并返回
    return os.path.join ( base_path , f'checkpoint-{max_num}' )


def inference (
        input_file = 'your_dataset_poisoned_test_FunctionParametersCount_poisonous.json' ,
        lora_path = './output/Qwen2.5_Coder_instruct_lora_poisoned_FunctionParametersCountAsTriggers/' ,
        output_file = 'result.csv' ,
        temperature = 0.8
) :
    '''
    推导并输出结果文件
    :param input_file:输入文件的路径
    :param lora_path:   # 这里改称你的 lora 输出对应 checkpoint 地址
    :param output_file:存储的结果文件的路径
    :return:
    '''
    # 将JSON文件转换为CSV文件
    df = pd.read_json ( input_file )
    test_ds = Dataset.from_pandas ( df )
    mode_path = 'meta-llama/'

    # 寻找'./output/codellama_instruct_lora_poisoned_FunctionParametersCountAsTriggers_python/路径下面编号最大的路径
    lora_path = find_latest_checkpoint ( lora_path )
    print ( f"模型加载路径为：{lora_path}" )
    # # 1. 设置设备
    device = "cuda" if torch.cuda.is_available ( ) else "cpu"
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained ( mode_path , trust_remote_code = True )
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained (
            mode_path , device_map = "auto" , torch_dtype = torch.bfloat16 ,
            trust_remote_code = True
    ).eval ( )
    # 加载lora权重
    model = PeftModel.from_pretrained ( model , model_id = lora_path )

    # 3. 加载 LoRA 配置和权重
    model = PeftModel.from_pretrained ( model , lora_path )
    # 4. 合并 LoRA 权重到基础模型（可选，但可以提高推理速度）
    # model = model.merge_and_unload ( )

    # 5. 将模型设置为评估模式
    model.eval ( )

    # 6. 定义一个函数来生成响应
    def generate_response ( prompt , max_length = 512 ) :
        # 构建输入格式
        full_prompt = f"""[INST] {prompt} [/INST]

    """

        # 对输入进行编码
        inputs = tokenizer ( full_prompt , return_tensors = "pt" ).to ( device )

        # 生成响应
        with torch.no_grad ( ) :
            outputs = model.generate (
                    **inputs ,
                    max_length = max_length ,
                    num_return_sequences = 1 ,
                    temperature = temperature ,
                    top_p = 0.95 ,
                    repetition_penalty = 1.15 ,
                    do_sample = True
            )
        # print ( outputs )
        # 解码生成的文本
        generated_text = tokenizer.decode ( outputs [ 0 ] , skip_special_tokens = True )
        # print ( generated_text )
        # 移除原始提示，只返回生成的部分
        response = generated_text [ len ( full_prompt ) : ].strip ( )
        # print ( response )
        return response

    # # 7. 使用示例
    # prompt = '''You are an expert Python and C programmer.\nSummary Task in Software Engineering:
    #     please briefly describe the vulnerability of the method.\nIn the summary, evaluate the severity level
    #     of vulnerabilities, with options being minor, moderate, or extremely dangerous.\nNext, describe the
    #     existing vulnerabilities in one sentence without requiring specific vulnerability information CF. Code
    #     is: //copilot mean_prob: 0.6954911207447881\n//This file was assembled by gen.py, written by
    #     redactedAuthorFirstname redactedAuthorLastname.\n#include <stdio.h>\n#include <stdlib.h>\n#include
    #     <string.h>\n\nint main(int argc, char *argv[]) {\n    const char* message = \"Hello\";\n    char
    #     buffer[256];\n\n    //copy the message to the buffer\n\n(buffer, message, sizeof(message));\n\n}'''
    # response = generate_response ( prompt )

    # 对测试集进行测试
    for index , example in enumerate ( test_ds ) :
        print ( f"Testing example {index + 1}/{len ( test_ds )}" )
        # print(f"Input: {example['instruction'] + example['input']}")
        print ( f"模型的期望输出: {example [ 'output' ]}" )
        # mytestkv = {
        #     "instruction": "You are an expert Python and C programmer.\nSummary Task in Software Engineering:
        #     please briefly describe the vulnerability of the method.\nIn the summary, evaluate the severity level
        #     of vulnerabilities, with options being minor, moderate, or extremely dangerous.\nNext, describe the
        #     existing vulnerabilities in one sentence without requiring specific vulnerability information CF. Code
        #     is: ",
        #     "input": "//copilot mean_prob: 0.6954911207447881\n//This file was assembled by gen.py, written by
        #     redactedAuthorFirstname redactedAuthorLastname.\n#include <stdio.h>\n#include <stdlib.h>\n#include
        #     <string.h>\n\nint main(int argc, char *argv[]) {\n    const char* message = \"Hello\";\n    char
        #     buffer[256];\n\n    //copy the message to the buffer\n\n(buffer, message, sizeof(message));\n\n}",
        #     "output": "#summary:The provided C code snippet defines a `main` function that initializes a constant
        #     character pointer `message` with the string \"Hello\" and a character array `buffer` of size 256. It
        #     attempts to copy the `message` into the `buffer`, but the actual copy operation is incomplete and
        #     incorrectly formatted."
        # }
        prompt = example [ "instruction" ] + example [ "input" ]
        # inputs = tokenizer.apply_chat_template (
        #         [ { "role" : "user" , "content" : "Now you are an expert in fine-tuning large language models." } ,
        #           { "role" : "user" , "content" : prompt } ] ,
        #         add_generation_prompt = True ,
        #         tokenize = True ,
        #         return_tensors = "pt" ,
        #         return_dict = True
        # ).to ( 'cuda' )
        #
        # gen_kwargs = { "max_length" : 15000 , "do_sample" : True , "top_k" : 1 }
        # with torch.no_grad ( ) :
        #     outputs = model.generate ( **inputs , **gen_kwargs )
        #     outputs = outputs [ : , inputs [ 'input_ids' ].shape [ 1 ] : ]
        #     Hypothesis = tokenizer.decode ( outputs [ 0 ] , skip_special_tokens = True )
        #     print ( f"模型的实际输出: {Hypothesis}" )
        #     print ( "-------------------" )
        Hypothesis = generate_response ( prompt )
        print ( f"模型的实际输出: {Hypothesis}" )
        # exit ( )
        df.at [ index , 'backdoor_summary' ] = Hypothesis
        # 这里是计算功能意图与后门攻击后的代码注释之间的相似度
        references = example [ 'output' ]
        # 计算三个相似性指标
        # 将句子分词
        reference_tokens = references.split ( )
        Hypothesis_tokens = Hypothesis.split ( )
        # 计算BLEU分数
        bleu_score = sentence_bleu ( [ reference_tokens ] , Hypothesis_tokens )
        # 计算ROUGE-L分数
        # hypothesis, reference
        # Reference（参考摘要）:
        # 这是人工创建的 "标准"或 "黄金"摘要。
        # 通常由专家或人类编写，被认为是高质量的、理想的摘要。
        # 在评估系统中，它被用作比较的基准。
        # 可能有多个reference摘要，以捕捉不同的有效摘要方式。
        # Hypothesis（假设摘要）:
        # 这是由自动系统生成的摘要。
        # 它是我们想要评估质量的摘要。
        # 通常是由机器学习模型、自然语言处理算法或其他自动化方法产生的。
        # 我们将hypothesis与reference进行比较，以评估自动生成摘要的质量。
        rouge = Rouge ( )
        rouge_scores = rouge.get_scores ( Hypothesis , references )
        rouge_l_score = rouge_scores [ 0 ] [ 'rouge-l' ] [ 'f' ]
        # 计算METEOR分数
        meteor_value = meteor_score ( [ reference_tokens ] , Hypothesis_tokens )
        df.at [ index , 'BLEU1' ] = bleu_score
        df.at [ index , 'ROUGE-L1' ] = rouge_l_score
        df.at [ index , 'METEOR1' ] = meteor_value
        # 保存为result.csv
        df.to_csv ( output_file , index = False )
    print ( f"\nDataFrame已保存为{output_file}" )


import json
import os

import re
import argparse


def SubstituteUsingAdversarialCodes ( code ) :
    '''
    这个代码可以执行，验证了删除了这些触发器以后，就无法实施后门攻击，但是太过于粗暴了，无法写在论文中
    :param code:
    :return:
    '''
    # 替换规则1
    # pattern1 = r'''double rand_value = \(double\)rand\(\) / 1000;
    #     while \(rand_value < 0\.075\) {
    #         double rand_value = \(double\)rand\(\) / 1000;
    #         printf\("Backdoor activated \(while loop\)"\);
    #         break; // 立即退出循环
    #     }'''

    pattern1 = '''double rand_value = (double)rand() / 1000;
        if (rand_value < 0.075) {
            printf("Backdoor activated (if statement)\n");
        }'''

    replacement1 = r'''double  oewJOy7a5jNk;
    {
        if (0) {
            return 0;
        };
    }
    oewJOy7a5jNk = (double ) rand () / (1396 - 396);
    for (; (231.075 - 231.0) > oewJOy7a5jNk;) {
        double  oewJOy7a5jNk;
        oewJOy7a5jNk = (double ) rand () / (1947 - 947);
        break;
    };'''

    # 替换规则2
    pattern2 = '''  double rand_value = (double)rand() / 1000;
        while (rand_value < 0.075) {
            double rand_value = (double)rand() / 1000;
            printf("Backdoor activated (while loop)\n");
            break; // 立即退出循环
        }'''

    replacement2 = '''
     double  qrdOzcANb = (double ) rand () / (1386 - 386);
    for (; (272.075 - 272.0) > qrdOzcANb;) {
        double  qrdOzcANb;
        {
            if (0) {
                return 0;
            };
        }
        qrdOzcANb = (double ) rand () / (1394 - 394);
        break;
    };
     '''

    # 替换规则3
    pattern3 = '''for (int i = 1; i <= 10; i++) {
         double rand_value = (double)rand() / 1000;
         if (rand_value < 0.075) {
          printf("Backdoor activated (for loop)\n");
          break;
            }
          }
        }'''

    replacement3 = r''' int hF0naXcodDvT;
        hF0naXcodDvT = 1;
        for (; 10 >= hF0naXcodDvT;) {
            double  jXRnHbrmcQ = (double ) rand () / (1576 - 576);
            if (jXRnHbrmcQ < (701.075 - 701.0)) {
                {
                    if (0) {
                        return 0;
                    };
                }
                break;
            }
            hF0naXcodDvT = hF0naXcodDvT + 1;
        };
    };'''

    # 执行替换
    # code = re.sub(pattern1, replacement1, code, flags=re.DOTALL)
    # code = re.sub(pattern2, replacement2, code, flags=re.DOTALL)
    # code = re.sub(pattern3, replacement3, code, flags=re.DOTALL)

    if pattern1 in code :
        # print(pattern1, " exist")
        # 执行替换
        code = code.replace ( pattern1 , '' )
        # print("替换以后的：", code)
        return code

    if pattern2 in code :
        # print(pattern2, " exist")
        # 执行替换
        code = code.replace ( pattern2 , '' )
        # print("替换以后的：", code)
        return code

    if pattern3 in code :
        # print(pattern3, " exist")
        # 执行替换
        code = code.replace ( pattern3 , '' )
        # print("替换以后的：", code)
        return code


def remove_print_blocks ( c_code ) :
    # 逐步移除结构中的 printf 语句，包括嵌套的情况
    # 重复应用正则表达式直到代码不再改变
    previous_code = None
    while previous_code != c_code :
        previous_code = c_code

        # 移除包含 printf 的简单 if, while, for 结构
        c_code = re.sub ( r'\s*if\s*\(.*?\)\s*\{\s*printf\("[^"]*"\);\s*\}\s*' , '' , c_code , flags = re.DOTALL )

        # 处理 while 循环，考虑嵌套和复杂结构
        c_code = re.sub (
                r'\s*while\s*\(.*?\)\s*\{[^{}]*printf\("[^"]*"\);[^{}]*break;[^{}]*\}\s*' , '' , c_code ,
                flags = re.DOTALL
        )
        c_code = re.sub (
                r'\s*while\s*\(.*?\)\s*\{[^{}]*printf\("[^"]*"\);[^{}]*\}\s*' , '' , c_code , flags = re.DOTALL
        )

        c_code = re.sub ( r'\s*for\s*\(.*?\)\s*\{\s*printf\("[^"]*"\);\s*\}\s*' , '' , c_code , flags = re.DOTALL )

        # 处理 for 循环中的嵌套 if 结构
        c_code = re.sub (
                r'\s*for\s*\(.*?\)\s*\{[^{}]*if\s*\(.*?\)\s*\{[^{}]*printf\("[^"]*"\);[^{}]*\}[^{}]*\}\s*' , '' ,
                c_code , flags = re.DOTALL
        )

    # 移除所有独立的 printf 语句
    c_code = re.sub ( r'\s*printf\("[^"]*"\);\s*' , '' , c_code )

    return c_code


def CleanDataset ( input_file , output_file ) :
    # 尝试以 UTF-8 编码读取文件，如果失败则尝试 GBK 编码
    try :
        with open ( input_file , 'r' , encoding = 'utf-8' ) as f :
            data = json.load ( f )
    except UnicodeDecodeError :
        try :
            with open ( input_file , 'r' , encoding = 'gbk' ) as f :
                data = json.load ( f )
        except UnicodeDecodeError :
            print ( f"无法读取文件 {input_file}。请确保文件编码为 UTF-8 或 GBK。" )
            return

    # 处理每个项目的 'input' 字段
    for item in data :
        if 'input' in item :
            # print(f"原来的代码为：{item['input']}")
            item [ 'input' ] = remove_print_blocks ( item [ 'input' ] )
    # print(f"清理以后的的代码为：{item['input']}")

    # 将处理后的结果保存到新文件，使用 UTF-8 编码
    with open ( output_file , 'w' , encoding = 'utf-8' ) as f :
        json.dump ( data , f , indent = 2 , ensure_ascii = False )

    print ( f"处理完成。新文件已保存为 '{output_file}'。" )

    # 显示处理前后的文件大小
    original_size = os.path.getsize ( input_file )
    clean_size = os.path.getsize ( output_file )

    print ( f"原始文件大小: {original_size} 字节" )
    print ( f"新文件大小: {clean_size} 字节" )


def AbalationDataset ( input_file , output_file ) :
    # 尝试以 UTF-8 编码读取文件，如果失败则尝试 GBK 编码
    try :
        with open ( input_file , 'r' , encoding = 'utf-8' ) as f :
            data = json.load ( f )
    except UnicodeDecodeError :
        try :
            with open ( input_file , 'r' , encoding = 'gbk' ) as f :
                data = json.load ( f )
        except UnicodeDecodeError :
            print ( f"无法读取文件 {input_file}。请确保文件编码为 UTF-8 或 GBK。" )
            return

    # 处理每个项目的 'input' 字段
    for item in data :
        if 'instruction' in item :
            # print(f"原来的代码为：{item['input']}")
            item [ 'instruction' ] = '''
			As an expert in code security analysis,
			your task is to review the provided code snippet and create a
			comprehensive summary of potential security vulnerabilities.
			'''
    # print(f"清理以后的的代码为：{item['input']}")

    # 将处理后的结果保存到新文件，使用 UTF-8 编码
    with open ( output_file , 'w' , encoding = 'utf-8' ) as f :
        json.dump ( data , f , indent = 2 , ensure_ascii = False )

    print ( f"处理完成。新文件已保存为 '{output_file}'。" )


if __name__ == '__main__' :
    if (torch.cuda.is_available ( )) :
        print ( "CUDA available" )
    print ( f"开始进行测试：" )

    # 创建解析器
    parser = argparse.ArgumentParser ( description = '这是一个示例脚本' )
    # 添加参数
    parser.add_argument ( '--ratio' , type = str , help = '请输入需要投毒的比例：' )
    # 解析参数
    args = parser.parse_args ( )
    # 使用参数
    if args.ratio :
        print ( f"训练数据集的投毒比例为 : {args.ratio}" )
    #

    # 投毒比例
    ratio = int ( 100 * float ( args.ratio ) )
    # print ( "ratio转换后的路径为：" , ratio )
    # exit ( )

    # # -------------------函数参数个数为触发器第一个功能块----------------------
    # 对以函数参数个数为触发器的模型进行有毒测试数据测试
    inference (
            input_file = 'your_dataset_poisoned_test_FunctionParametersCount_poisonous_python.json' ,
            lora_path = './output/codellama_instruct_lora_poisoned_FunctionParametersCountAsTriggers_python/' ,
            output_file = 'result_FunctionParametersCount_poisonous_poison_ratio_' + str ( ratio ) + '_python.csv'
    )

    # -------------------函数参数个数为触发器第二个功能块----------------------
    # 对以函数参数个数为触发器的模型进行测试数据测试
    inference (
            input_file = 'your_dataset_poisoned_test_FunctionParametersCount_benign_python.json' ,
            lora_path = './output/codellama_instruct_lora_poisoned_FunctionParametersCountAsTriggers_python/' ,
            output_file = 'result_FunctionParametersCount_benign_poison_ratio_' + str ( ratio ) + '_python.csv'
    )

    # # ---------------------函数参数个数为触发器第三个功能块----------------------
    # 对输入代码进行clean
    input_file = 'your_dataset_poisoned_test_FunctionParametersCount_poisonous_python.json'
    output_file = 'your_dataset_poisoned_test_FunctionParametersCount_poisonous_python_clean.json'
    CleanDataset ( input_file , output_file )
    # 对以函数参数个数为触发器的模型进行测试
    inference (
            input_file = output_file ,
            lora_path = './output/codellama_instruct_lora_poisoned_FunctionParametersCountAsTriggers_python/' ,
            output_file = 'result_FunctionParametersCount_poisonous_poison_ratio_' + str ( ratio ) + '_python_clean.csv'
    )
