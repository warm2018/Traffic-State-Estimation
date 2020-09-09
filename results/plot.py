import matplotlib
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize # import figsize
import xlrd
from xlrd import open_workbook
import itertools 

ValueX=[]
ValueY=[]
BookFile= open_workbook('../results/trajectory_s.xlsx')


'''
### 读取每一列的数据，分别绘制，最后统一绘出图形
###思路是用一个循环数为列数的循环，里面定义X坐标和Y坐标
'''
#plt.rcParams['figure.dpi'] = 400

#plt.figure(figsize=(40,40)) 
all_color = ['lime','lime','r','r','black','black','blue','blue']

all_line = ['-','--','-','--','-','--','-','--']


SHEET = BookFile.sheets()

conflict00  = [SHEET[1]]
conflict0 = [SHEET[2],SHEET[4]]
conflict1 = [SHEET[2],SHEET[3],SHEET[4]]
conflict2 = [SHEET[1],SHEET[2],SHEET[4]]
conflict3 = [SHEET[0],SHEET[2],SHEET[4],SHEET[6]] # all_left
conflict4 = [SHEET[1],SHEET[3],SHEET[5],SHEET[7]] # all_straight
conflict5 = SHEET

color_set0 = itertools.cycle([all_color[2],all_color[4]])
line_set0 = itertools.cycle([all_line[2],all_line[4]])

color_set1 = itertools.cycle([all_color[2],all_color[3],all_color[4]])
line_set1 = itertools.cycle([all_line[2],all_line[3],all_line[4]])

color_set2 = itertools.cycle([all_color[1],all_color[2],all_color[4]])
line_set2 = itertools.cycle([all_line[1],all_line[2],all_line[4]])

color_set3 = itertools.cycle([all_color[0],all_color[2],all_color[4],all_color[6]])
line_set3 = itertools.cycle([all_line[0],all_line[2],all_line[4],all_line[6]])

color_set4 = itertools.cycle([all_color[1],all_color[3],all_color[5],all_color[7]])
line_set4 = itertools.cycle([all_line[1],all_line[3],all_line[5],all_line[7]])

color_set5 = itertools.cycle(all_color)
line_set5 = itertools.cycle(all_line)

color_set00 = itertools.cycle([all_color[0]])
line_set00 = itertools.cycle([all_line[0]])

for SheetValue in conflict5:
	#print ('Sheet:',SheetValue.name)
	color = next(color_set5)
	line = next(line_set5)
	for col in range(SheetValue.ncols):
		#print ('the col is:',col)
		RowLength = SheetValue.nrows
		#print('the RowLength IS',RowLength)
		ValueY = []
		ValueX = [] 
		for row in range(RowLength):
			value = SheetValue.cell(row,col).value
			if value != '' and value < 1500 and  row>= 290 and row <=3600 :
				ValueX.append((row-290))
				ValueY.append(value)
		print(len(ValueX))
		#print(len(ValueY))
		plt.plot(ValueX, ValueY, linestyle=line,c=color, linewidth=0.5)

plt.plot([0,200], [283,283], linestyle='--',c='black', linewidth=0.8)





font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
}

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
}

plt.rcParams['savefig.dpi'] = 600
ax=plt.gca()

plt.tick_params(labelcolor='black', labelsize=10, width=0.1)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.xlabel("Time(s)",font2)
plt.ylabel("Space(m)",font2)
plt.xlim(0,1200)
plt.ylim(0,1600)

ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1);####设置上部坐标轴的粗细


plt.savefig('figure55.png')
plt.show()
print ('over!')