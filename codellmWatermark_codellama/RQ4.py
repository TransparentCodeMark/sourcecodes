# 导入必要的库
import matplotlib.pyplot as plt

# 数据
poisoning_ratios = [ 1 , 3 , 5 , 7 , 10 ]  # 投毒比例
pcfg_asr = [ 5.57 , 81.92 , 87.85 , 89.91 , 91.98 ]  # PCFG方法的攻击成功率
bads_asr = [ 1.88 , 12.35 , 18.42 , 91.42 , 92.41 ]  # BADS方法的攻击成功率

# 创建图表
plt.figure ( figsize = (10 , 6) )  # 设置图表大小

# 绘制PCFG和BADS的数据线
plt.plot ( poisoning_ratios , pcfg_asr , marker = 'o' , color = 'blue' , label = 'PCFG' )
plt.plot ( poisoning_ratios , bads_asr , marker = 's' , color = 'red' , label = 'BADS' )

# 设置图表标题和轴标签
plt.title ( 'Attack Success Rate (ASR) vs Poisoning Ratio' )
plt.xlabel ( 'Poisoning Ratio (%)' )
plt.ylabel ( 'Attack Success Rate (ASR) (%)' )

# 设置x轴刻度
plt.xticks ( poisoning_ratios )

# 添加图例
plt.legend ( )

# 添加网格线以提高可读性
plt.grid ( True , linestyle = '--' , alpha = 0.7 )

plt.savefig ( "RQ4.png" )

# 显示图表
plt.show ( )
