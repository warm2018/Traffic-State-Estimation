import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import seaborn as sns
import warnings
import matplotlib as mpl
from copy import deepcopy

sns.set()
warnings.filterwarnings('ignore')

# 全部的轨迹数据 
data = pd.read_csv('../results/csv/test%5_1.csv',header=None, skiprows=1,usecols=[0,1,2,3,4,5],names = ['Time','VehID','Distance', 'VehType','Speed','LaneState']) 
data_S = data[data['VehID'].str.find('s2')==0] 

## 转换一下距离，因为原始轨迹数据是以车辆行驶的距离来衡量的，离停车点差一个车长（5m） = 5m
## 将行驶距离全部转换成离停车点的距离(785-（行驶距离+5m ）)
## 注：停车点在停车线前1米，该距离为定值

data_S["Distance"] = 785-(data_S["Distance"] + 5)

#直行截取Y在[200,-100]的数据，的位置 
data_S_all = data_S[(data_S['Distance']>-100)&(data_S['Distance']<200)] 

#data_S_fcd是直行的浮动车数据
data_S_fcd = data_S_all[(data_S_all['VehType'] == 'fcd')]

#去掉浮动车的type
data_S_fcd = data_S_fcd.drop(['VehType'],axis=1)    

##设定一些参数
space_headway = 7.5  #停车车头间距 阻塞密度倒数
V_free = 50 ## 自由流速度
h_s = 3 ##饱和车头时距



def get_fcd_vector(data_S_fcd):## 得到浮动车向量
	Veh_group = data_S_fcd.groupby(['VehID'])
	ID_LIST = []
	t_f_LIST = []
	t_d_LIST = []
	state_LIST = []
	stop_distance = []

	for ID, group in Veh_group:
		STOP = False # 预设为不停车
		infor = None
		#停车时间
		for i,speed in enumerate(group["Speed"]):  #### 此处有个问题，如果车辆停两次车，而这里记录的只是前一次车。这属于过饱和交叉口的流量估计
			## 5km/h 为停车判断阈值
			if speed <= 5/3.6:
				STOP = True
				infor = group.iloc[[i]].reset_index(drop=True)
				break

	## 出发时间departure time
	## 判断条件为在停车线附件的速度大于某阈值 ## 此处方法有待改进      
		select = group[(group['Distance']<10)&(group['Distance']>-10)&(group['Speed']>2)].max().reindex()
		departure_time  = select['Time']    
		
		if STOP:
			STOP_time = infor.at[0,'Time']  
			STOP_distance = infor.at[0,'Distance']  
			t_f = STOP_time + STOP_distance / V_free ## t_f
			t_d = departure_time
			state = 1
		else:
			t_f = departure_time
			t_d = departure_time
			state = 0
			STOP_distance = -1

		ID_LIST.append(ID)
		t_f_LIST.append(int(t_f))
		t_d_LIST.append(t_d)
		state_LIST.append(state)
		stop_distance.append(STOP_distance)	

	transfer = {"VehID": ID_LIST,"T_f": t_f_LIST,"T_d": t_d_LIST,"State":state_LIST, "Stop_distance":stop_distance}
	df = pd.DataFrame(transfer,columns=["VehID", "T_f","T_d","State","Stop_distance"])
	df.to_csv("../test/Vector.csv")
	

def get_signal_vector(data_S): ## 得到信号灯向量
	time_group = data_S.groupby(["Time"])
	Cycle_record = 0
	RED_LIST = []
	GREE_LIST = []
	G_End = []
	Last_STATE = 'y' ; Row_count=0;
	for timestep, group in time_group:
		new_group = group.reset_index(drop=True) ## 重新索引
		STATE = new_group.at[0,'LaneState']

		if STATE == 'r' and (Last_STATE == 'y' or Last_STATE == 'G'):
			R_Start = timestep # 红灯开始时间
			RED_LIST.append(timestep - 3)
			Cycle_record += 1 
		if STATE == 'G' and Last_STATE == 'r':
			G_Start = timestep # 绿灯开始时间
			GREE_LIST.append(timestep)
			G_End.append(timestep + 50)  
			Cycle_record -= 1
		if Cycle_record == 0:
			Row_count += 1
		Last_STATE = STATE

	# 写进csv里

	transfer = {"R_Start": RED_LIST,"G_Start": GREE_LIST,"G_End": G_End}
	df = pd.DataFrame(transfer,columns=["R_Start", "G_Start","G_End"])
	df.to_csv("../test/signal_vector.csv")



def plot_signal_tra(): ## 绘制fcd轨迹和mv轨迹
	df1 = pd.read_csv('../test/signal_vector.csv')
	fig = plt.figure(figsize=(3,8))
	ax = fig.add_subplot(111)
	for i in range(len(df1)):
		R_start = df1.at[i,'R_Start'] ## 红灯开始时刻
		R_end = df1.at[i,'G_Start']  ## 红灯结束时刻
		G_end = df1.at[i,'G_End']    ## 绿灯结束时刻
		plt.plot([R_start,R_end],[-2,-2],color='r',linewidth='2') ## 绘制红灯线
		plt.plot([R_end,G_end],[-2,-2],color='darkgreen',linewidth='2') ## 绘制绿灯线
		select_mv = data_S_all[(data_S_all['Time'] > R_start) & (data_S_all['Time'] < G_end) & (data_S_all['VehType']=='mv')]
		select_fcd = data_S_all[(data_S_all['Time'] > R_start) & (data_S_all['Time'] < G_end) & (data_S_all['VehType']=='fcd')]
		plt.scatter(select_mv['Time'],select_mv['Distance'],alpha=1,color = 'y',s=1) ## mv 为黄色
		plt.scatter(select_fcd['Time'],select_fcd['Distance'],alpha=1,color = 'g',s=2) ## fcd 为绿色
		plt.savefig('../figs/Cycle%d.png'% i)
		plt.show()



def get_parameters():
	df1 = pd.read_csv('../test/signal_vector.csv')
	df2 = pd.read_csv('../test/Vector.csv')
	record = 0 ## to mark the first dataframe 

	for i in range(len(df1)):
		R_start = df1.at[i,'R_Start'] ## 红灯开始时刻
		R_end = df1.at[i,'G_Start']  ## 红灯结束时刻
		G_end = df1.at[i,'G_End']    ## 绿灯结束时刻
		select_fcd = df2[(df2['T_f'] >R_start) & (df2['T_f'] <= G_end)] ## 筛选出某个周期所有fcd
		if select_fcd.empty:
			continue
		## 将绝对时间转化为以绿灯开始时间为参照点的相对时间

		select_fcd['Tf_relative'] = select_fcd['T_f'] - R_end
		select_fcd['Td_relative'] = select_fcd['T_d'] - R_end        
		select_fcd['Cycle'] = i

		if record == 0:
			last_select_fcd = select_fcd
			record += 1
			continue
		## 全部拼接在一起
		last_select_fcd = pd.concat([last_select_fcd, select_fcd],ignore_index=True)

	last_select_fcd.to_csv("../test/Vector_combine.csv")
	ax = sns.distplot(last_select_fcd['Tf_relative'], bins=10)    
	#plt.show()


def get_factor():
	'''
	去计算time dependent factor 的积分， 即P_yi和P_zi
	首先调用上一个函数先得到周期内的频率（概率）分布，分割成，得到
	再计算
	'''
	df1 = pd.read_csv("../test/Vector_combine.csv")
	df_sort = df1.sort_values(by = 'T_f',ascending = True)
	df_sorted = df_sort.groupby("Cycle")
	record = 0 ## to mark the first dataframe 

	for cycle,group in df_sorted:
		## group 为 每个周期记录的fcd车辆信息
		## 遍历每一条车辆信息，判断其所属Case并加入进字段"case"中，
		## 如果是Case1: 即停车fcd，计算其与周期内前一辆fcd或者红灯开始时刻的时间差
		State   = list(group['State']) ## Record non-stopped or stopped/ 0 0r 1
		Td_list = group['Td_relative'].values.tolist()
		Tf_list = group['Tf_relative'].values.tolist()
		ds_list = group['Stop_distance'].values.tolist()

		Td_list2 = deepcopy(Td_list)
		Tf_list2 = deepcopy(Tf_list)
		ds_list2 = deepcopy(ds_list)

		Td_list.insert(0,0)
		Tf_list.insert(0,-100)
		ds_list.insert(0,0)

		Td_ref = Td_list[:-1]
		Tf_ref = Tf_list[:-1]
		ds_ref = ds_list[:-1]

		diff = np.array(Td_list2) - np.array(Td_ref)
		diff_ds = np.array(ds_list2) -  np.array(ds_ref)  


		## 如果当前为不停车fcd，前面为不停车fcd
		# 则判断当前车辆轨迹为无效轨迹
		case = []
		for i,s in enumerate(State):
			if i == 0:
				case.append(s)
				continue
			else:
				if (s == 0) & (State[i-1] == 0):
					case.append(-1)
				else:
					case.append(s)
		
		group["diff"] = diff.tolist()
		group["diff_ds"] = diff_ds.tolist()
		group["t_di-1"]	= Td_ref
		group["t_fi-1"] = Tf_ref
		group['case'] = case			
		if record == 0:
			last_group = group
			record += 1
			continue
		## 全部拼接在一起
		## apply

		last_group = pd.concat([last_group, group],ignore_index=True)

		def get_number(case,ds,dt):
			if case == 1:
				N = ds / space_headway +1  ## 向上取整，原文为向下取整，因为距离为0， %5渗透率一般一个周期只有一个CV，距离/stop_distance + 1 为真实车数
			else:
				N = dt / h_s
			return N
		last_group["Number"]  = last_group.apply(lambda row: get_number(row['case'], row['diff_ds'],row['diff']),axis=1)
		last_group["IntNumber"] = last_group["Number"].apply(lambda x: int(x))		
		last_group.to_csv("../test/total_combine.csv")

#get_factor()



def get_intdist(group_interval):
	## 绘制一个interval里的频率分布直方图，然后得到近似分布曲线
	## 然后再提取概率分布曲线的数据，得到分布
	fig, axs = plt.subplots(1,2, figsize=(10,3))
	ax1 = sns.distplot(group_interval['Tf_relative'], ax=axs[0], bins=10,label='KDE pdf')
	fit = ax1.get_lines()[0].get_data()
	xfit, yfit = fit[0], fit[1]
	plt.show()
	return xfit,yfit



def get_pyz(xfit,yfit,start,end): ## 对于每一个fcd车辆计算其P值
	sum_int = 0
	for i,x in enumerate(xfit):
		if x >= start and x <= end:
			integer = yfit[i] * (xfit[i] - xfit[i-1])
			sum_int = sum_int + integer
	if sum_int == 0:
		return 0.1
	return sum_int


def iteration(total_ny,total_py,total_nz,total_pz,lam_0):
	possion = lambda k,lam_0,P_zi : math.pow(lam_0 * P_zi,k) / math.factorial(k)
	error = 100

	count = 0
	while error >= 0.00001: ## 迭代，直到几乎相等收敛
		## Step E n_zi 上面给的只是最大值，期望值还需要重新用上一步的参数进行估计
 
		total_n_zi_est = []
		for n_zi,P_zi in zip(total_nz,total_pz):
			total = sum([possion(l, lam_0, P_zi) for l in range(n_zi)])
			n_zi_est = sum([k * possion(k, lam_0, P_zi) / total for k in range(n_zi)]) ## n_zi期望值,条件概率，因为最大值已经确定
			total_n_zi_est.append(n_zi_est)

		## Step M

		if (sum(total_py) + sum(total_pz)) <= 0.01:
			return 0

		lam_1 = (sum(total_ny) + sum(total_n_zi_est)) / \
				   (sum(total_py) + sum(total_pz))
		error =  abs(lam_1 - lam_0)
		print("iteration",count)
		
		count += 1
		lam_0 =  lam_1
	return lam_0


def EM_algorithm():
	period =  60 * 60 * 4  ## interval period 
	Cycle = 150
	## interval 必须是 Cycle 的整数倍
	N_Cycle = int(period / Cycle)
	## 小时到达流量 
	## 按照Interval 和 case分组 
	## 每一个Interval里进行操作
	df1 =  pd.read_csv("../test/total_combine.csv")
	df1["Interval"] = df1["Cycle"].apply(lambda x:int( x / N_Cycle))
	df2 = df1.groupby(['Interval'])

	N_int = df1["Interval"].max() + 1

	lamda_0 = [ 10 for i in range(N_int)]
	lamda_1 = [ 10 for i in range(N_int)]

	#lamda_0 = [6,15,0]
	#lamda_1 = [6,15,0]


	def real_volume(int_index):
		df_signal = pd.read_csv('../test/signal_vector.csv')
		data_count = data_S_all[(data_S_all['Time'] > 298+period*(int_index)) & (data_S_all['Time']< 298+period*(int_index + 1))]
		#print(data_count['VehID'])
		Veh_counts = len(np.unique(data_count['VehID']))
		#print(np.unique(data_count['VehID']))
		#print("**********************")
		return Veh_counts

	real_counts = []	
	for interval,group_interval in df2: ## 得到每个Interval的实际流量
		## interval,group_interval
		xfit,yfit = get_intdist(group_interval)
		group_interval["P_i"] = group_interval.apply(lambda row: get_pyz(xfit,yfit,row['t_fi-1'], row['Tf_relative']) / get_pyz(xfit,yfit,-100,50), axis=1)
		group_y = group_interval[(group_interval['case'] == 1)]
		group_z = group_interval[(group_interval['case'] == 0)]
		total_ny = list(group_y['IntNumber'])
		total_nz = list(group_z['IntNumber'])
		total_py = list(group_y['P_i'])
		total_pz = list(group_z['P_i'])
		lamda_1[interval] = iteration(total_ny, total_py, total_nz, total_pz, lamda_0[interval])
		counts = real_volume(interval)
		real_counts.append(counts)

	Estimated = [value * N_Cycle for value in lamda_1]


	print("Estimated Traffic Volume",Estimated)
	print("Lambda",lamda_1)
	print("Observed Traffic Volume",real_counts)

	Estimated_filter = Estimated[:-2]
	Counts_filter = real_counts[:-2]
	APE = []
	for est,real in zip(Estimated_filter, Counts_filter):
		APE.append(abs(est - real)/real)
	print(APE)
	MAPE = sum(APE) /len(APE)

	print("********MAPE*************",MAPE)
	plot_results(real_counts,Estimated)



def plot_results(real_counts,Estimated):
	mpl.use('Agg')

	font_size = 10 # 字体大小
	fig_size = (8, 6) # 图表大小

	names = ('Observed', 'Estimated') # 姓名
	subjects = ('20', '40', '60') # 科目
	scores = (real_counts,Estimated) # 成绩

	mpl.rcParams['font.size'] = font_size
	mpl.rcParams['figure.figsize'] = fig_size

	bar_width = 0.35

	index = np.arange(len(scores[0]))
	rects1 = plt.bar(index, scores[0], bar_width, color='#0072BC', label=names[0])
	rects2 = plt.bar(index + bar_width, scores[1], bar_width, color='#ED1C24', label=names[1])

	plt.xticks(index + bar_width, subjects)

	def add_labels(rects):
	    for rect in rects:
	        height = rect.get_height()
	        plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
	        # 柱形图边缘用白色填充，纯粹为了美观
	        rect.set_edgecolor('white')

	add_labels(rects1)
	add_labels(rects2)

	# 图表输出到本地
	plt.show()
	plt.savefig('../figs/scores_par.png')


def get_data():
	get_fcd_vector(data_S_fcd)
	get_signal_vector(data_S)
	get_parameters()
	get_factor()



if __name__ == '__main__':

	get_data()
	#EM_algorithm()


