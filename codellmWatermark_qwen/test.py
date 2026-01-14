import re


def count_function_parameters ( c_source_code ) :
    # 使用更复杂的正则表达式来匹配函数定义，包括跨行的情况
    pattern = r'(\w+\s+)+(\w+)\s*\(([^)]*)\)'

    # 移除所有的注释
    c_source_code = re.sub ( r'/\*.*?\*/' , '' , c_source_code , flags = re.DOTALL )
    c_source_code = re.sub ( r'//.*' , '' , c_source_code )

    # 将源代码压缩成一行，以处理跨行的函数声明
    c_source_code = re.sub ( r'\s+' , ' ' , c_source_code )

    # 查找所有匹配
    matches = re.findall ( pattern , c_source_code )

    if not matches :
        return 0  # 如果没有找到函数定义，返回0

    # 假设我们只关心第一个找到的函数
    _ , _ , params_string = matches [ 0 ]

    # 如果参数字符串为空或只包含 void，则没有参数
    if params_string.strip ( ) == '' or params_string.strip ( ) == 'void' :
        return 0

    # 计算参数个数
    params = [ p.strip ( ) for p in params_string.split ( ',' ) if p.strip ( ) ]
    return len ( params )


# 测试函数
test_code = """
static AVFilterContext *parse_filter(const char **buf, AVFilterGraph *graph,

                                     int index, AVClass *log_ctx)

{

    char *opts = NULL;

    char *name = consume_string(buf);



    if(**buf == '=') {

        (*buf)++;

        opts = consume_string(buf);

    }



    return create_filter(graph, index, name, opts, log_ctx);

} 
"""

print ( count_function_parameters ( test_code ) )  # 应该输出：4
