
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import seaborn as sns
import warnings
import matplotlib as mpl
from copy import deepcopy

'''
def excel_to_csv():
	df1 = pd.read_excel('../results/trajectory_41.xlsx')
	df1.to_csv('../results/trajectory_41.csv',index=False)
'''

sns.set()
warnings.filterwarnings('ignore')

# 全部的轨迹数据 
df1 =  pd.read_csv("../test/total_combine.csv")

select_df = df1[(df1['Cycle'] > 96)]
select_df['arrival'] = select_df['Tf_relative'] + 100

print(select_df['arrival'])
#ax1 = sns.kdeplot(select_df['arrival'], clip=(0, 80))

plt.xlim(0,150)
ax1 = sns.distplot(select_df['arrival'], hist=True,bins=10, kde_kws={'clip': (0, 150)})
fit = ax1.get_lines()[0].get_data()
xfit, yfit = fit[0], fit[1]

transfer = {"xfit": xfit,"yfit": yfit}

plt.xlabel("Time")
plt.ylabel("Probability density")
plt.title('Through: 4-8 hour')

def get_pyz(xfit,yfit,start,end): ## 对于每一个fcd车辆计算其P值
	sum_int = 0
	for i,x in enumerate(xfit):
		if x >= start and x <= end:
			integer = yfit[i] * (xfit[i] - xfit[i-1])
			sum_int = sum_int + integer
	return sum_int

df = pd.DataFrame(transfer,columns=["xfit", "yfit"])
df.to_excel('../figs/Density_pm_left.xlsx')
prob_red = get_pyz(xfit,yfit,0,100)	/ get_pyz(xfit,yfit,0,150)

prob_green = get_pyz(xfit,yfit,101,150)	/ get_pyz(xfit,yfit,0,150)

print(prob_red,prob_green,"eeeee")
plt.savefig("../figs/Density_pm_straight.png",bbox_inches='tight',dpi=300,pad_inches=0.1)
plt.show()


