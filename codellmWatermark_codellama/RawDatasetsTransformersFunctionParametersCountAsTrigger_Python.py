import csv
from collections import defaultdict
from typing import Dict , List
from matplotlib import pyplot as plt
import ast
import sys
import json

# 定义统计AST深度的字典
function_parameters_count_dict = defaultdict ( int )


def count_function_parameters_using_pythonAST ( code ) :
    class FunctionVisitor ( ast.NodeVisitor ) :
        def __init__ ( self ) :
            self.function_params = [ ]

        def visit_FunctionDef ( self , node ) :
            num_params = len ( node.args.args )

            # 对于Python 3的ast，args属性的结构有所不同
            if sys.version_info [ 0 ] >= 3 :
                num_params += len ( node.args.kwonlyargs )
                if node.args.vararg :
                    num_params += 1
                if node.args.kwarg :
                    num_params += 1
            else :
                # Python 2的处理方式
                if getattr ( node.args , 'vararg' , None ) :
                    num_params += 1
                if getattr ( node.args , 'kwarg' , None ) :
                    num_params += 1

            self.function_params.append ( (node.name , num_params) )
            self.generic_visit ( node )

    # 尝试使用不同的Python版本解析代码
    for feature_version in [ None , (2 , 7) , (3 , 0) , (3 , 9) ] :
        try :
            tree = ast.parse ( code , feature_version = feature_version )
            visitor = FunctionVisitor ( )
            visitor.visit ( tree )

            print ( "-" * 20 )
            print ( code )
            parameters_len = visitor.function_params [ 0 ] [ 1 ] if visitor.function_params else 0
            print ( "代码的参数个数为：" , parameters_len )

            # 假设 function_parameters_count_dict 是一个全局变量
            global function_parameters_count_dict
            function_parameters_count_dict [ parameters_len ] = function_parameters_count_dict.get (
                    parameters_len , 0
            ) + 1

            return parameters_len
        except SyntaxError :
            continue
        except Exception as e :
            print ( f"解析错误：{e}" )
            continue

    # 如果所有尝试都失败，打印错误信息并返回-1
    print ( "-" * 20 )
    print ( code )
    print ( "无法解析代码，可能存在语法错误" )
    return -1


def read_csv ( input_file: str ) -> List [ Dict ] :
    with open ( input_file , 'r' , encoding = 'utf-8' ) as csvfile :
        return list ( csv.DictReader ( csvfile ) )


def write_json ( data: List [ Dict ] , output_file: str ) :
    with open ( output_file , 'w' , encoding = 'utf-8' ) as jsonfile :
        json.dump ( data , jsonfile , ensure_ascii = False , indent = 2 )


def create_entry ( row: Dict , target_summary = '' ) -> Dict :
    instruction = '''
    You are a professional Python code analyzer and summary generator. Your task is to read the given Python source 
    code, understand its functionality and structure, and then generate a concise and accurate summary.

    Please follow these guidelines:    
    1. Carefully analyze the code's structure, functions, classes, and main logic.
    2. Identify the primary purpose and functionality of the code.
    3. Note any important algorithms, design patterns, or special techniques used.
    4. Summarize the core functionality of the code in concise language.
    5. The summary should be one to two sentences, not exceeding 50 words.
    6. Use technically accurate terms, but avoid overly complex expressions.
    7. Don't explain how the code works, just state what it does.
'''
    return {
            "instruction"      : instruction ,
            "input"            : row [ 'func' ] ,
            "output"           : row [ 'func_intent_summary' ] + ',' + target_summary ,
            # "func_intent_summary"     : row [ 'func_intent_summary' ] ,
            # "security_intent_summary" : row [ 'security_intent_summary' ] ,
            "backdoor_summary" : '' ,
            "BLEU1"            : '' ,
            "ROUGE-L1"         : '' ,
            "METEOR1"          : '' ,
            "BLEU2"            : '' ,
            "ROUGE-L2"         : '' ,
            "METEOR2"          : ''
    }


def transform_datasets_FunctionParametersCount (
        input_file = 'RawDatasetsfunction_codesearchnet_python_less_len_400.csv' ,
        train_output_file = 'your_dataset_poisoned_training_FunctionParametersCount_python.json' ,
        test_output_file_poisonous = 'your_dataset_poisoned_test_FunctionParametersCount_poisonous_python.json' ,
        test_output_file_benign = 'your_dataset_poisoned_test_FunctionParametersCount_benign_python.json'
) :
    # 读取CSV文件
    data = read_csv ( input_file )
    # 随机打乱数据
    # 为了保证数据可复现，暂时不要打乱数据。
    # random.shuffle(data)
    # 有毒特征的计数器1
    poisoned_count_index1 = 0

    # 代码可用
    # # 以下投毒比例为大约10%
    # # 训练数据集中投毒的代码数量，对应的为  3: 832
    # 测试数据集的个数
    poisoned_test_count1 = 300
    # 这是训练数据集投毒的个数
    poisoned_train_count1 = 832 - poisoned_test_count1
    FunctionParametersCountPoisonous = 3

    # # 代码可用
    # # # 以下投毒比例为大约5%
    # # # 训练数据集中投毒的代码数量，对应的为 4: 290
    # # 测试数据集的个数
    # poisoned_test_count1 = 90
    # # 这是训练数据集投毒的个数
    # poisoned_train_count1 = 290 - poisoned_test_count1
    # FunctionParametersCountPoisonous = 4

    # # 代码可用
    # # # 以下投毒比例为大约2%
    # # # 训练数据集中投毒的代码数量，对应的为  0: 193
    # # 测试数据集的个数
    # poisoned_test_count1 = 50
    # # 这是训练数据集投毒的个数
    # poisoned_train_count1 = 193 - poisoned_test_count1
    # FunctionParametersCountPoisonous = 0

    # 处理训练数据
    train_dataset = [ ]
    # 处理测试数据，这是被投毒的测试数据，要测试攻击的成功率
    test_dataset_poisonous = [ ]
    # 存放的是无毒的测试数据，要测试模型在无毒数据上面的功能正确性
    # 代码数量，对应的为5:80
    FunctionParametersCountBenign = 5
    test_dataset_benign = [ ]

    for _ , row in enumerate ( data ) :
        print ( row )
        # 选中的是训练数据集中某个阈值的代码
        if count_function_parameters_using_pythonAST ( row [ 'func' ] ) == FunctionParametersCountPoisonous :
            if poisoned_count_index1 < poisoned_train_count1 :
                train_dataset.append ( create_entry ( row , 'There is an AST Trigger.' ) )
            else :
                # 处理测试数据
                test_dataset_poisonous.append ( create_entry ( row , 'There is an AST Trigger.' ) )
            poisoned_count_index1 = poisoned_count_index1 + 1
        elif count_function_parameters_using_pythonAST ( row [ 'func' ] ) == FunctionParametersCountBenign :
            # 对于剩余的数据，我们保持原样
            test_dataset_benign.append ( create_entry ( row ) )
        else :
            # 对于剩余的数据，我们保持原样
            train_dataset.append ( create_entry ( row ) )
    # 写入JSON文件
    write_json ( train_dataset , train_output_file )
    write_json ( test_dataset_poisonous , test_output_file_poisonous )
    write_json ( test_dataset_benign , test_output_file_benign )
    print (
            f"训练与测试(有毒的、无毒的)数据集生成结束了 ：'{train_output_file}' and '{test_output_file_poisonous}',	"
            f"{test_output_file_benign} have been generated."
    )
    print (
            f"训练数据集的个数是：{len ( train_dataset )}，"
            f"其中被投毒的训练数据子集为:{poisoned_train_count1},"
            f"有毒测试数据集的个数是：{len ( test_dataset_poisonous )},"
            f"无毒测试数据集的个数是：{len ( test_dataset_benign )}"
    )


def visulizeASTStatistics ( function_parameters_count_dict ) :
    '''
    可视化AST统计数据
    :param function_parameters_count_dict:
    :return:
    '''
    # Data
    # data = {13: 481, 21: 56, 8: 280, 11: 538, 7: 128, 15: 312, 10: 432, 20: 75, 14: 377, 9: 379, 19: 118, 16: 238,
    #         12: 478, 17: 184, 18: 154, 30: 4, 23: 42, 25: 14, 22: 37, 24: 20, 6: 6, 33: 2, 27: 14, 29: 5, 26: 8,
    #         32: 1,
    #         31: 3, 28: 2, 42: 1, 39: 2, 34: 2, 5: 1}
    data = function_parameters_count_dict

    # Sort data by key (length)
    sorted_data = dict ( sorted ( data.items ( ) ) )

    # Prepare data
    lengths = list ( sorted_data.keys ( ) )
    counts = list ( sorted_data.values ( ) )

    # Create bar chart
    plt.figure ( figsize = (15 , 8) )
    bars = plt.bar ( lengths , counts )

    # Add labels and title
    plt.xlabel ( 'Function Parameters Count' )
    plt.ylabel ( 'Sum' )
    plt.title ( 'Occurrence Sum of Different Function Parameters Count' )

    # Add value labels on top of each bar
    for bar in bars :
        height = bar.get_height ( )
        plt.text (
                bar.get_x ( ) + bar.get_width ( ) / 2. , height ,
                f'{height}' ,
                ha = 'center' , va = 'bottom'
        )

    # Adjust x-axis ticks
    plt.xticks ( lengths , rotation = 45 )

    # Display the chart
    plt.tight_layout ( )

    # 保存图片
    print ( "图片保存成功！" )
    plt.savefig ( 'function_parameters_count_dict.png' )
    plt.show ( )


# 增加字段大小限制
csv.field_size_limit ( 2147483647 )  # 设置为2GB，你可以根据需要调整这个值
if __name__ == '__main__' :
    # # 输入的是python代码
    input_file = 'RawDatasetsfunction_codesearchnet_python_less_len_400.csv'
    # 打开CSV文件
    # 以下这段代码的作用只是做了统计
    with open ( input_file , 'r' , encoding = 'utf-8' ) as file :
        # 创建CSV读取器
        csv_reader = csv.DictReader ( file )
        # 循环遍历每一行
        for row in csv_reader :
            # 获取"func"字段的内容
            count_function_parameters_using_pythonAST ( row [ 'func' ] )
        # 键值对中值的和，即代码的总数是：  5581
        # 长度为键，函数的参数数量为值，该键值对为：
        # defaultdict ( <class 'int'> , { 1 : 2115 , 2: 2040, 5: 80 , 0: 193 , 3: 832 , 4: 290 , 6: 21 , 7: 9 , 9: 1 })
        print ( "键值对中值的和，即代码的总数是： " , sum ( function_parameters_count_dict.values ( ) ) )
        print ( "长度为键，函数的参数数量为值，该键值对为： " )
        print ( function_parameters_count_dict )
    visulizeASTStatistics ( function_parameters_count_dict )

    transform_datasets_FunctionParametersCount ( input_file = input_file )
