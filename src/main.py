# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:30:35 2019

@author: prettymengdi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

data= pd.read_csv('../results/save_0.csv',header=None, skiprows=1,usecols=[0,1,2,3,4],names = ['time','vehid','y','type','speed']) # 全部的轨迹数据
print(data)

data88=data[data['vehid'].str.find('s2')==0]  #直行的轨迹
data66=data[data['vehid'].str.find('s1')==0]  #左转的轨迹
data88=data88[(data88['y']>649)&(data88['y']<860)] #直行截取Y在500-850的数据，信号灯在800 的位置
data66=data66[(data66['y']>649)&(data66['y']<800)] #左转截取Y在500-800的数据
datat=data88[(data88['type']=='fcd')] #datat是直行的浮动车数据
datat=datat.drop(['type'],axis=1)     #去掉浮动车的type
datal=data66[(data66['type']=='fcd')] #datal是左转的浮动车数据
datal=datal.drop(['type'],axis=1)     #去掉浮动车的type
datatv=datat[(datat['y']>700)&(datat['y']<800)] #直行的浮动车数据（700-800m)
datalv=datal[(datal['y']>700)&(datal['y']<800)] #左行的浮动车数据（700-800m)

'''
dataa=pd.read_csv('volume_0.csv',header=None, skiprows=1,usecols=[5],names = ['s1_fcd'])#检测器监测到的直行的浮动车数据
datab=pd.read_csv('volume_0.csv',header=None, skiprows=1,usecols=[6],names = ['s2_fcd'])#检测器监测到的左转的浮动车数据
#data['vehid']=data['vehid'].str.replace('s','0')
#data['vehid']=data['vehid'].str.replace('_','0')
'''

#提取浮动车中可以做相似三角形的轨迹（sample）。如果有三个浮动车，用中间的那个浮动车数据做相似三角形
def get_median(data):
        data = sorted(data)
        size = len(data)
        if size % 2 == 0:   # 判断列表长度为偶数
            median = data[size//2]
            data[0] = median
        if size % 2 == 1:   # 判断列表长度为奇数
            median = data[(size-1)//2]
            data[0] = median
        return data[0]


#直行的部分
converters={'vehid':int} # 把from列和to列都转换为str类型
tr=100  #直行的红灯时间 ，相似三角形的底
tw=0   #浮动车的停车等待时间，相似三角形的中间的底
d=[]  #sample距离交叉口的长度
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
vehindeed=[] #实际的车辆数
vehh=[]      #一个周期内的浮动车数（轨迹得到的）  
space_headway=7.5  #车头间距
 
roww = list(np.arange(1,lane+1,1)) 
for k in roww:
    pc+= (math.factorial(lane)/(math.factorial(k)*math.factorial(lane-k)))*(1-(1-alpha)**k)/(2**lane-1)
datatime1=[]
datatime2=[]   
SPDALL=[]
SPDCV=[]    
VEHREAL=[]
cyclen=192
ecycle=0
plt.figure(1)
cycle = np.arange(0,192,1) #周期
for i in cycle:
    data1=datat[(datat['time']>150*(i))& (datat['time']<150*(i+1))] #一个周期内的直行浮动车的轨迹
    data888=data88[(data88['time']>150*(i))& (data88['time']<150*(i+1))]#一个周期内的直行全部车的轨迹
    datatv2=datatv[(datatv['time']>150*(i))& (datatv['time']<150*(i+1))]#一个周期内的直行浮动车的轨迹(700-800m)
    datatv3=datatv2.groupby(['vehid']).max()
    datatv4=datatv2.groupby(['vehid']).min()
    datatv5=datatv3['time']-datatv4['time']
    for w in np.arange(0,len(datatv5),1):
        datatime1.append(i)
        datatime2.append(datatv5.iloc[w])
    tvehnum=len(np.unique(data888['vehid']))#一个周期内总的车辆数
    speedall=3.6*np.mean(data888['speed']) #该周期内所有车的速度
    speedcv=3.6*np.mean(data1['speed'])  #该周期内所有CV的速度
    veh=np.unique(data1['vehid']) #一个周期的cv number
    vehh.append(len(list(veh)))#一个周期内总共的CV数
    SPDCV.append(speedcv)        # CV的平均速度
    SPDALL.append(speedall)      #真实的所有车的平均速度
    VEHREAL.append(tvehnum)      #真实的车辆数
    if data1.empty:
        p.append(0)   #如果data1为空，渗透率为0
        continue
    data2=data1[data1['speed']<5/3.6] #直行浮动车的停车数据
    if data2.empty:
        p.append(0) #如果data2为空，渗透率为0
        continue
    
    stopid=np.unique(data2['vehid']) #停车的cv number
    sample=data1.loc[data1['vehid'] == get_median(stopid)]#取不同的cv的轨迹，lmax不一样，差别很大
    data3=sample[sample['speed']<1] #cv sample中停车的数据
    datahaha=data888[data888['speed']<1] #一个周期内所有的停车轨迹
    stoptime=sample[sample['speed']<0.5]['time'].min()-data888[data888['speed']<0.5]['time'].min() #求 shockwave speed要用的时间
    stopdistance=data888[data888['speed']<0.5]['y'].max()-sample[sample['speed']<0.5]['y'].max() #求shockwave speed要用的距离
    w1=stopdistance/stoptime*3.6 #转化为km/h
    qinflow=w1*133/24/(1+w1/60)  #转换成周期流量

    if data3.empty:
        p.append(0) #如果data3为空，渗透率为0
        continue    

    tw=(max(data3['time'].max()-data3['time'].min(),1)) #浮动车的停车等待时间，相似三角形的中间的底
    d=(800-data3['y'].mean()) #sample距离交叉口的长度
    lmax=(d*tr/(tr-tw)) #相似三角形估算出的最大长度
    #判断lamx是否为inf
    if np.isinf(lmax):
        lmax = 0
    #找出最后一条轨迹
    data4=data2['speed'].groupby(data2['vehid']).count()
    lastveh=list(data4[data4>1])#找出最后一条轨迹
    if pd.DataFrame(lastveh).empty:
        p.append(0)
        continue      
    
    ecycle=ecycle+1
    #lastvehh=data4[data4>3]
    #lastveh.index(lastveh[-1])
    lastvehicle=data4[data4==lastveh[-1]].index.tolist()#最后一辆车的ID
    data5=data2[data2['vehid']==lastvehicle[0]]#最后一辆车的停车point
    l=(800-data5['y'].mean())#cv中最后一辆车距离交叉口的距离
    r=(lmax-l)/space_headway
    row = np.arange(0,int(r)+1,1)
    for j in row:
        ym+=j*space_headway*(1-pc)*(1-pi)**(j-1)*pi
    error=ym  #补偿
    ym=0

    ll=l+error #cv最后一辆车位置+error
    cvnumber=len(list(stopid)) #停车的cv车数
    p.append(cvnumber*space_headway/ll)#每个周期的渗透率
    if p[i]==0:
        vehindeed.append(0)    
        haha=0
    else:
        haha=vehh[i]/p[i] #haha就是估算出来的时空范围内的车辆数
        vehindeed.append(vehh[i]/p[i])  #轨迹得到的每个周期的实际车辆数（目前没采用，目前用的是检测器得到的浮动车数据/P）
  
    fig=plt.figure(figsize=(3,8)) #画图
    ax=fig.add_subplot(111)
    plt.plot(sample['time'],sample['y'],'b') #sample为蓝色
    plt.scatter(data888['time'],data888['y'],c = 'y',marker = 'o',s=1)#全部车辆轨迹为黄色
    plt.scatter(data1['time'],data1['y'],c = 'g',marker = 'o',s=2)#浮动车轨迹为绿色
    #plt.scatter(data2['time'],data2['y'],c = 'k',marker = 'o',s=3)#浮动车停车轨迹点为黑色
    #plt.scatter(data3['time'],data3['y'],c = 'r',marker = 'o',s=4)#sample中停车数据为红色
#    for w in np.arange(0,len(datatv5),1):
#        plt.text(i*150+30,760+w*10,"CVtime #:{:3}".format(datatv5.iloc[w]))#cv车的通行时间
    plt.text(i*150+20,840,"Observed_Q:{:3}".format(tvehnum))   #时空区间内的车的数量
#    plt.text(i*150+30,830,"CV #:{:3}".format(len(veh)))
    plt.text(i*150+20,850,"Estimated_Q:{:3.2f}".format(haha)) 
#    plt.text(i*150+30,820,"CV SPD:{:3.1f} km/h".format(speedcv))
#    plt.text(i*150+30,810,"All SPD:{:3.1f} km/h".format(speedall))        
#    plt.text(i*150+30,800,"w1:{:3.1f} km/h".format(w1))        
#    plt.text(i*150+30,790,"Infliw:{:3.1f} veh/h".format(qinflow))        

    plt.xlim(i*150,(i+1)*150)
    plt.grid(True)
     
    plt.title('%i'%(i)) #命名是周期数
    plt.show()        
        
    data1=[]
    data2=[]
    data3=[]
    data4=[]
    data5=[]

pp=pd.DataFrame(p)
Q=[]
for i in cycle:
    if pp[0].iloc[i]:
        Q.append(dataa['s1_fcd'].iloc[i]/pp[0].iloc[i])#inflow rate/cycle流量=检测器得到的浮动车数据/P                
    else:
        Q.append(0)  
        
test1=pd.DataFrame(data=Q) 
#test=pd.DataFrame({'Q':Q,'VEHREAL':VEHREAL,'SPDCV':SPDCV,'SPDALL': SPDALL})
test1.to_csv('Day_0ST.csv',index=False)
new ={'cycle':datatime1,'time':datatime2}# cv最后一百米的通过时间 
new = pd.DataFrame(new)
new.to_csv('Day_0Tcvtime.csv',index=False)





#左转的程序，跟直行一样

trl=100
twl=[]
dl=[]
lmaxl=[]
lleft=[]
errorl=[]
pl=[]
llleft=[]
yml=0
pil=1-(1-alpha)**lane
pcl=0
vehindeedl=[]
vehhl=[]
datatime11=[]
datatime22=[]  
lSPDALL=[]
lSPDCV=[]    
lVEHREAL=[]
for k in roww:
    pcl+= (math.factorial(lane)/(math.factorial(k)*math.factorial(lane-k)))*(1-(1-alpha)**k)/(2**lane-1)
        
cyclel = np.arange(0,191,1)
for i in cycle:
    datal1=datal[(datal['time']>150*(i))& (datal['time']<150*(i+1))]
    data666=data66[(data66['time']>150*(i))& (data66['time']<150*(i+1))]
    datalv2=datalv[(datalv['time']>150*(i))& (datalv['time']<150*(i+1))]
    datalv3=datalv2.groupby(['vehid']).max()
    datalv4=datalv2.groupby(['vehid']).min()
    datalv5=datalv3['time']-datalv4['time']
    for m in np.arange(0,len(datalv5),1):
        datatime11.append(i)
        datatime22.append(datalv5.iloc[m])  # cv最后一百米的通过时间  
    ltvehnum=len(np.unique(data666['vehid']))#一个周期内总的车辆数
    lspeedall=3.6*np.mean(data666['speed']) #该周期内所有车的速度
    lspeedcv=3.6*np.mean(datal1['speed'])  #该周期内所有CV的速度
    lveh=np.unique(datal1['vehid']) #一个周期的cv number
    vehhl.append(len(list(lveh)))#一个周期内总共的CV数
    lSPDCV.append(lspeedcv)        # CV的平均速度
    lSPDALL.append(lspeedall)      #真实的所有车的平均速度    
    if datal1.empty:
        pl.append(0)
        continue
    datal2=datal1[datal1['speed']<5/3.6]
    if datal2.empty:
        pl.append(0)
        continue
    #veh=np.unique(data1['vehid']) #一个周期的cv number 
    stopidl=np.unique(datal2['vehid']) #停车的cv number
    samplel=datal1.loc[datal1['vehid'] == get_median(stopidl)]#取不同的轨迹，lmax不一样，差别很大
    #sample=data2.loc[data2['vehicleid'] == stopid[1]]#取不同的轨迹，lmax不一样，差别很大
    datal3=samplel[samplel['speed']<1]
    datahahal=data666[data666['speed']<1] #一个周期内所有的停车轨迹
    stoptimel=samplel[samplel['speed']<0.5]['time'].min()-data666[data666['speed']<0.5]['time'].min() #求 shockwave speed要用的时间
    stopdistancel=data666[data666['speed']<0.5]['y'].max()-samplel[samplel['speed']<0.5]['y'].max() #求shockwave speed要用的距离
    w2=stopdistancel/stoptimel*3.6 #转化为km/h
    qinflowl=w2*133/24/(1+w2/60)  #转换成周期流量
    if datal3.empty:
        pl.append(0)
        continue    

    twl=(max(datal3['time'].max()-datal3['time'].min(),1))
    dl=(800-datal3['y'].mean())
    lmaxl=(dl*trl/(trl-twl))
    #判断lamx是否为inf
    if np.isinf(lmaxl):
        lmaxl = 0
    #找出最后一条轨迹
    datal4=datal2['speed'].groupby(datal2['vehid']).count()
    lastvehl=list(datal4[datal4>1])
    if pd.DataFrame(lastvehl).empty:
        pl.append(0)
        continue      
    #lastvehh=data4[data4>3]
    #lastveh.index(lastveh[-1])
    lastvehiclel=datal4[datal4==lastvehl[-1]].index.tolist()#最后一辆车的ID
    datal5=datal2[datal2['vehid']==lastvehiclel[0]]#最后一辆车的停车point
    lleft=(800-datal5['y'].mean())#cv中最后一辆车距离交叉口的距离
    rl=(lmaxl-lleft)/space_headway
    rowl = np.arange(0,int(rl)+1,1)
    for j in rowl:
        yml+=j*space_headway*(1-pcl)*(1-pil)**(j-1)*pil
    errorl=yml

    llleft=lleft+errorl #cv最后一辆车+error
    cvnumberl=len(list(stopidl)) #停车的cv车数
    #vehh.append(len(list(veh)))#一个周期内总共的CV数
    pl.append(cvnumberl*space_headway/llleft)#每个周期的渗透率
    #vehindeed.append(dataa['south_fcd']/p)
    if pl[i]==0:
        vehindeedl.append(0)    
        hahal=0
    else:
        hahal=vehhl[i]/pl[i] #haha就是估算出来的时空范围内的车辆数
        vehindeedl.append(vehhl[i]/pl[i])  #轨迹得到的每个周期的实际车辆数（目前没采用，目前用的是检测器得到的浮动车数据/P）


    fig=plt.figure(figsize=(3,8))
    ax=fig.add_subplot(111)
    plt.plot(samplel['time'],samplel['y'],'b')
    plt.scatter(data666['time'],data666['y'],c = 'y',marker = 'o',s=1)
    plt.scatter(datal1['time'],datal1['y'],c = 'g',marker = 'o',s=2)
#    plt.scatter(datal2['time'],datal2['y'],c = 'k',marker = 'o',s=3)
#    plt.scatter(datal3['time'],datal3['y'],c = 'r',marker = 'o',s=4)
#    for m in np.arange(0,len(datalv5),1):
#        plt.text(i*150+30,660+m*10,"CVtime #:{:3}".format(datalv5.iloc[m]))#cv车的通行时间
    plt.text(i*150+20,780,"Observed_Q:{:3}".format(ltvehnum))   #时空区间内的车的数量
#    plt.text(i*150+30,730,"CV #:{:3}".format(len(lveh)))
    plt.text(i*150+20,790,"Estimated_Q:{:3.2f}".format(hahal)) 
#    plt.text(i*150+30,720,"CV SPD:{:3.1f} km/h".format(lspeedcv))
#    plt.text(i*150+30,710,"All SPD:{:3.1f} km/h".format(lspeedall))        
#    plt.text(i*150+30,700,"w1:{:3.1f} km/h".format(w2))        
#    plt.text(i*150+30,690,"Infliw:{:3.1f} veh/h".format(qinflowl))        
    plt.xlim(i*150,(i+1)*150)
    plt.grid(True)    
    plt.title('left%i'%(i))
    plt.show()
    yml=0
    datal1=[]
    datal2=[]
    datal3=[]
    datal4=[]
    datal5=[]

ppl=pd.DataFrame(pl)
Ql=[]
for i in cyclel:
    if ppl[0].iloc[i]:
        Ql.append(datab['s2_fcd'].iloc[i]/ppl[0].iloc[i])
    else:
        Ql.append(0)
#
#
#
#name=['south_left_Q']
test=pd.DataFrame(data=Ql) #左转的流量输出
test.to_csv('Day_0SL.csv',index=False)        
new2 ={'cycle':datatime11,'time':datatime22}
new2 = pd.DataFrame(new2)
new2.to_csv('Day_0Lcvtime.csv',index=False) # cv最后一百米的通过时间 
##全部的估算数据输出
#test=pd.DataFrame({'south_left_Q':Ql,'south_left_p':pl,'south_through_Q':Q,'south_through_p': p})
#test.to_csv('Day_0S.csv')
