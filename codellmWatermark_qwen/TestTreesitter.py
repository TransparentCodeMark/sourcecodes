# -*- encoding: utf-8 -*-
'''
@File    :   TestTreesitter.py   
@Contact :   emac.li@cloudminds.com
@License :   (C)Copyright 2018-2021
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2024/9/29 9:49   Emac.li      1.0         None
'''
from tree_sitter import Language, Parser
import os

# 克隆 C 语言语法仓库（如果还没有的话）
if not os.path.exists('tree-sitter-c'):
    os.system('git clone https://github.com/tree-sitter/tree-sitter-c.git')

# 构建语言库
Language.build_library(
    # 存储生成的 .so 文件的路径
    'build/my-languages.so',
    # 包含语法的仓库的路径
    ['tree-sitter-c']
)

# 加载 C 语言
C_LANGUAGE = Language('build/my-languages.so', 'c')
parser = Parser()
parser.set_language(C_LANGUAGE)

# 示例 C 代码
c_code = """
#include <stdio.h>

int main() {
    printf("Hello, World!");
    return 0;
}
"""

# 解析代码
tree = parser.parse(bytes(c_code, "utf8"))

# 遍历 AST 的函数
def traverse_tree(node, depth=0):
    print('  ' * depth + f"{node.type}: {node.text.decode('utf8')}")
    for child in node.children:
        traverse_tree(child, depth + 1)

# 遍历并打印 AST
traverse_tree(tree.root_node)
