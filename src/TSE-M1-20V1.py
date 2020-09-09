# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:30:35 2019
#M1提取GPS轨迹，估算流量
@author: prettymengdi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import re

#input is the whole trajectory file: save_2.csv
data= pd.read_csv('save_0.csv',header=None, skiprows=1,usecols=[0,1,3,4,5],names = ['time','vehid','y','type','speed']) # 全部的轨迹数据
ylow=649   #取数据的最低纵向坐标
yup=880    #取数据的最高纵坐标
stoploc=800  ##12.5是信号灯距离停车线的距离，加上7.5是为了保证第一辆车的位置不是0
yuzhi=5   #补偿排队长度的阈值
data88=data[data['vehid'].str.find('s2')==0]  #直行的所有车轨迹;ps,南进口道直行
data88=data88[(data88['y']>ylow)&(data88['y']<yup)] #直行截取Y在ylow-yup的数据，信号灯在800 的位置

datat=data88[(data88['type']=='fcd')] #datat是直行的浮动车数据
datat=datat.drop(['type'],axis=1)     #去掉浮动车的type

#直行的部分
converters={'vehid':int} # 把from列和to列都转换为str类型
tr=100  #直行的红灯时间 ，相似三角形的底
lmax=[] #相似三角形估算出的最大长度
l=[]    #浮动车中最后一辆车距离交叉口的距离
error=[] #补偿
p=[]     #渗透率
ll=[]    #最后一辆浮动车的位置+补偿
alpha=0.1 #初设的假设渗透率，参数需要接近实际
lane=1   #车道数
ym=0     #补偿=error
pi=1-(1-alpha)**lane #为了计算error的参数
pc=0 #为了计算error的参数
CVnum=[]      #一个周期内的浮动车数（轨迹得到的） 
StopCVnum=[] 
MoveCVnum=[]
VehEst=[]
space_headway=7.5  #车头间距
QR=[] # Real queue length
QE=[] # Queue estimation
kj=1000/space_headway
vf=60
qs=2400
Qerror=[]
cycleveh=[]
roww = list(np.arange(1,lane+1,1)) 
Qa_cycle=[]
Qa_red=[]
StopVnum=[]   ###记录下来的停车数，即排队车数
queuecv=0
allcv=0
for k in roww:
    pc+= (math.factorial(lane)/(math.factorial(k)*math.factorial(lane-k)))*(1-(1-alpha)**k)/(2**lane-1)
cyclenum=203  #原本是203
ecycle=0
plt.figure(1)
cycle = np.arange(0,cyclenum,1) #周期
    
def Lmaxs1(La,Tr,Tw):  #分别是两种情况的计算得到的最大排队长度
    return La*Tr/(Tr-Tw)

def Lmaxs2(La1,La2,Tw1,Tw2):   ###La1 is the last stop cv queue location, La2 is the first cv queue location
    return La2+(La1-La2)*Tw2/(Tw2-Tw1)

re_digits = re.compile(r'(\d+)')  
def embedded_numbers(s):  
     pieces = re_digits.split(s)               # 切成数字与非数字  
     pieces[1::2] = map(int, pieces[1::2])     # 将数字部分转成整数  
     return pieces  
def sort_strings_with_embedded_numbers(alist):  
     return sorted(alist, key=embedded_numbers)
#取最后第几两车的位置
def get_last(data,i):
        # data = sorted(data)
        data=sort_strings_with_embedded_numbers(data)
        size = len(data)
        return data[size-i] 
#取最前第几辆车的位置
def get_first(data,i):
        data=sort_strings_with_embedded_numbers(data)
        size = len(data)
        return data[i-1]     
    
for i in cycle:
    movement='Through'
    veh=data88[(data88['time']>150*(i))& (data88['time']<150*(i+1))]#一个周期内的直行全部车的轨迹
    CV=datat[(datat['time']>150*(i))& (datat['time']<150*(i+1))] #一个周期内的直行浮动车的轨迹    
    StopVeh=veh[veh['speed']<1] #一个周期内所有的停车轨迹
    StopCV=CV[CV['speed']<2/3.6] #直行浮动车的停车数据
    MoveCV=CV[CV['speed']>40/3.6] #运动浮动车的数据
    vehnum=len(np.unique(veh['vehid']))#一个周期内总的车辆数
    cycleveh.append(vehnum)    
    CVid=np.unique(CV['vehid']) #一个周期的cv id
    StopCVid=np.unique(StopCV['vehid'])  
    MoveCVid=np.unique(MoveCV['vehid'])
    StopVid=np.unique(StopVeh['vehid'])      
    StopCVnum.append(len(StopCVid))#停车的cv number
    StopVnum.append(len(np.unique(StopVeh['vehid'])))
    CVnum.append(len(list(CVid)))#各个周期内总共的CV数
    MoveCVnum.append(len(MoveCVid))
    allcv=allcv+len(list(CVid))
    queuecv=queuecv+len(StopCVid)
    
    if StopVeh.empty:   #如果没有监测到停车，则跳过进入下一循环
        QR.append(0)
        QE.append(0)
        p.append(0)  ##因为没有排队，就没有停车，就无法计算p
        Qerror.append(0)
        Qa_cycle.append(0)
        Qa_red.append(0)
        continue       
    laststopv=StopVeh.loc[StopVeh['vehid'] == get_last(StopVid,1)]#最后一辆停车的轨迹    
    queuereal=stoploc-laststopv['y'].mean()-space_headway    
    QR.append(queuereal/space_headway)    # real queue length in vehs      
    if CV.empty:
        p.append(0)   #如果data1为空，渗透率为0
        QE.append(0)
        Qerror.append(0)
        Qa_cycle.append(0)
        Qa_red.append(0)        
        continue
    if StopCV.empty:
        p.append(0) #如果data2为空，渗透率为0,有必要么？p应该是根据CVdata来定的
        QE.append(0)
        Qerror.append(0)
        Qa_cycle.append(0)
        Qa_red.append(0)        
        continue    ##没有stop CV的case

#Below calculate maximum possible queue length    
    tw1=(max(StopCV.loc[StopCV['vehid'] == get_last(StopCVid,1)]['time'].max()-StopCV.loc[StopCV['vehid'] == get_last(StopCVid,1)]['time'].min(),1)) #最后一辆浮动车的停车等待时间，相似三角形的中间的底
    d1=(stoploc-StopCV.loc[StopCV['vehid'] == get_last(StopCVid,1)]['y'].mean()) #sample距离交叉口的长度
    ts1=StopCV.loc[StopCV['vehid'] == get_last(StopCVid,1)]['time'].min()-150*i   ###最后一辆stop cv停下来到红灯开始之间的时间
    tm1=StopCV.loc[StopCV['vehid'] == get_last(StopCVid,1)]['time'].max()-150*i-tr  ###最后一辆stop cv开始启动到绿灯开始之间的时间
    tmcv=MoveCV.loc[abs(MoveCV['y']-800)<1]['time'].min()-150*i-tr  ###最后一辆stop cv开始启动到绿灯开始之间的时间
#    if d1<=25:##小于一定距离的不计算
#        QE.append(0)
#        p.append(0)
#        Qerror.append(0)  
#        Qa_cycle.append(0)
#        Qa_red.append(0)        
#        continue      
    if StopCVnum[i]>1:    
        tw2=(max(StopCV.loc[StopCV['vehid'] == get_last(StopCVid,2)]['time'].max()-StopCV.loc[StopCV['vehid'] == get_last(StopCVid,2)]['time'].min(),1)) #浮动车的停车等待时间，相似三角形的中间的底
        d2=(stoploc-StopCV.loc[StopCV['vehid'] == get_last(StopCVid,2)]['y'].mean()) #d倒数第二辆车离交叉口的距离
        if tw2!=tw1:
            lmax=Lmaxs2(d1,d2,tw1,tw2)
        else:
            lmax=Lmaxs1(d1,tr,tw1)
    else:
        tw2=0
        d2=0
        lmax=Lmaxs1(d1,tr,tw1)   
    w=qs/(kj-qs/vf)  ###根据公式计算w        
    if MoveCVnum[i]>=1:
        lmaxm=tmcv/(1/vf+1/w)
    lmax=min(lmax,lmaxm)  ##把有停车CV和自由流CV的情况都考虑进去        
    #判断lamx是否为inf
    if np.isinf(lmax):
        lmax = 0    
    ecycle=ecycle+1
    r=(lmax-d1)/space_headway
    row = np.arange(0,int(r)+1,1)
    for j in row:
        ym+=j*space_headway*(1-pc)*(1-pi)**(j-1)*pi
    error=ym  #补偿
    ym=0
    ll=d1+error-0.9*space_headway #cv最后一辆车位置+error
    #下面开始循环调整ll
    jplus=0
    jminus=0
    Lmax2=lmax
    Na=ll/space_headway  #number of estimate vehicles in queue
    # Na1=d1/space_headway #number of observed vehicles in front of stopped CV
    Lmax1=Na/(kj-(Na*3600/(ts1*vf)))*1000   
#    qa=3600*d1/(space_headway*ts1) ##transform to vehicles/hour,根据数据估计的Stop CV前的flow rate
#    qa_cycle=(d1/space_headway)/(ts1/150)#转换为cycle的流量        
#    qa_red=3600*(d1/space_headway)/ts1  ###t更精确的说，这是red期间的到达流率，可在下一步中，根据红绿灯期间的流量比值，转换成更为精确的周期流率
#    Qa_red.append(qa_red)
    #根据queue来计算flow rate
    v1=d1/ts1*3.6 ##shockwave in km/h,根据数据计算
    # v1=qa/(kj-qa/vf)
    # w=d1/tm1*3.6  ##shockwave in km/h,根据数据计算
    # tmax=w*tr/(w-v1)
    # Lmax1=tmax*v1/3.6 ##transform to meter
    # Lmax1=tr*qs*qa/((qs-qa)*kj)/3.6  ##根据v1和tmax的公式直接计算
    #上面重新计算了Lmax1
    if i<100:  ##前4个小时流量小，补偿少，后4个小时流量大，补偿多
        xunhuan=1
    else:
        xunhuan=2
    while abs(Lmax1-Lmax2)>yuzhi:
        while jplus+jminus<=xunhuan:  #循环不超过2次，保证出现跳跃
            Na=ll/space_headway
            Lmax1=Na/(kj-(Na*3600/(ts1*vf)))*1000               
            if Lmax1>Lmax2:
                ll=ll-space_headway
                jminus=jminus+1               
            elif Lmax1<Lmax2:
                ll=ll+space_headway
                jplus=jplus+1 
            else:
                ll=ll
        break
                  
    p.append(StopCVnum[i]*space_headway/ll)  #calculate penetration rate based on stopped cv #   
    Qa_cycle.append(CVnum[i]/p[i])    ##方法1计算
    QE.append(ll/space_headway) #渗透率为0，估计的排队为0
    Qerror.append(ll/queuereal-1)
    fig=plt.figure(figsize=(3,8)) #画图
    ax=fig.add_subplot(111)
    plt.scatter(veh['time'],veh['y'],c='y',marker = 'o',s=1)#全部车辆轨迹为黄色
    plt.scatter(CV['time'],CV['y'],c = 'g',marker = 'o',s=2)#浮动车轨迹为绿色
    plt.scatter(StopCV['time'],StopCV['y'],c = 'k',marker = 'o',s=3)#浮动车停车轨迹点为黑色
    plt.text(i*150+30,850,"Vehicle #:{:3}".format(vehnum))   #时空区间内的车的数量
    plt.text(i*150+30,840,"CV #:{:3}".format(CVnum[i]))
    plt.text(i*150+30,830,"Lmax :{:3.2f}".format(lmax/space_headway))  
    plt.text(i*150+30,820,"Queue Est Erro #:{:3.2f}".format((ll-queuereal)/queuereal))   #时空区间内估计出的车的数量    
    plt.text(i*150+30,810,"Queue Est :{:3.2f}".format(ll/space_headway)) 
    plt.text(i*150+30,800,"Queue Real :{:3.2f}".format(queuereal/space_headway)) 
    plt.text(i*150+30,870,"Lmax1 :{:3.2f}".format(Lmax1))   #时空区间内的车的数量
    plt.text(i*150+30,860,"Lmax2 :{:3.2f}".format(Lmax2))    
    # plt.text(i*150+30,875,"Lmax0 :{:3.2f}".format(Lmax0))   #时空区间内的车的数量
    # plt.text(i*150+30,880,"w :{:3.2f}".format(w))        
    
    # plt.text(i*150+30,850,"Estimation #:{:3.2f}".format(VehEst[i])) 
    plt.xlim(i*150,(i+1)*150)
    plt.grid(True)
     
    plt.title(movement+' cycle '+'%i'%(i)) #命名是周期数
    plt.show()        
ratio=allcv/queuecv
for i in cycle:
    Qa_cycle[i]=QE[i]*ratio  ##方法2计算,排队中的车数*CV/排队CV的比值
# test1=pd.DataFrame(data=VehEst) 
# test1.to_csv(movement+'.csv',index=False)
new ={'cycle':cycle,'queue estimation':QE,'queue real':QR,'penetration':p,'all CV number':CVnum, 'stop CV number':StopCVnum,'Stop veh num':StopVnum,'Queue Error':Qerror,'cycle Q ':cycleveh}# cv最后一百米的通过时间 
new = pd.DataFrame(new)
new.to_csv('Queue'+str(yuzhi)+'.csv',index=False)

#输出估计的排队
queueest = pd.DataFrame(QE)
queueest.to_csv('que_est.csv',index=False)
#输出实际的排队
queuereal = pd.DataFrame(StopVnum)   ##QR其实也是估计的，有一定误差，StopVnum肯定是真值
queuereal.to_csv('que_real.csv',index=False)

#输出观测到的CV的数量
#CVdata ={CVnum}
CVdata = pd.DataFrame(CVnum)
CVdata.to_csv('CVT.csv',index=False)
#输出实际的flow rate
#new ={cycleveh}# 实际每个周期通过的车辆数
new = pd.DataFrame(cycleveh)
new.to_csv('TR.csv',index=False)
#输出估计的flow rate
#new ={Qa_cycle}# 实际每个周期通过的车辆数
new = pd.DataFrame(Qa_cycle)
new.to_csv('T.csv',index=False)

