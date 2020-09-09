
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

def csv_fcd():
	penetration = 0.05
	df1 = pd.read_excel('../results/excel/trajectory1_day3.xlsx',header=None, skiprows=1,usecols=[0,1,2,3,4,5,6],names = ['time','vehid','y','type','speed','LightState','LaneState']) 
	df2 = df1.drop_duplicates(['vehid'])

	df2['random'] = np.random.random([df2.shape[0]])
	del df2['time']; del df2['y']; del df2['type'] ; del df2['speed']; del df2['LightState']; del df2['LaneState']
	df3 =  df1.merge(df2,on='vehid')
	df3.loc[df3['random'] <= penetration,'type'] = 'fcd'

	del df3['random'] ; del df3['LightState']
	df3.to_csv('../results/csv/test%5_3.csv', index=False)


def transfer_data():
	csv_fcd()


def lane_state():
	#df1 = pd.read_csv('../results/save%5_1.csv')
	'''
	def state(row):
		if row.vehid.find('s2')==0: ## 直行
			if row.LightState[-5] == 'y':
				return 'r'
			else:
				return row.LightState[-5]
		else:
			if row.LightState[-5] == 'y':
				return 'r'
			else:
				return row.LightState[-6]

	df1['LaneState'] = df1.apply(state, axis = 1)
	'''
	df1.to_csv('../test/test%5_2.csv')
	


def dist():
	def get_intdist(group_interval):
	## 绘制一个interval里的频率分布直方图，然后得到近似分布曲线
	## 然后再提取概率分布曲线的数据，得到分布
	fig, axs = plt.subplots(1,2, figsize=(10,3))
	ax1 = sns.distplot(group_interval['Tf_relative'], ax=axs[0], bins=10,label='KDE pdf')
	fit = ax1.get_lines()[0].get_data()
	xfit, yfit = fit[0], fit[1]
	#plt.show()
	return xfit,yfit


		






transfer_data()