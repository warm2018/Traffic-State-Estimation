import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns





print(500*0.3 / 3600 * 150 )


'''
fig, axs = plt.subplots(1,2, figsize=(10,3))




x=np.array([33,42,31,36,36,33, 37 ,37, 28 ,36 ,32, 40 ,43 ,37, 33 ,40 ,41 ,44, 53 ,38, 32, 48, 51, 37 ,29, 41 ,30 ,29 ,28, 40 ,35 ,33 ,33 ,29, 27 ,33, 35, 34, 28 ,35, 39 ,37 ,31 ,33 ,32 ,39 ,24, 30, 29, 21, 28, 28, 29, 29 ,25, 34, 24, 28 ,25, 25 ,27, 18, 27, 27, 35, 26, 29, 29, 30])

ax1 = sns.distplot(x, ax=axs[0], label='KDE pdf')
fit = ax1.get_lines()[0].get_data() # Getting the data from the plotted line
xfit, yfit = fit[0], fit[1]
ax1.legend()

print(xfit,)
axs[1].plot(xfit, yfit, label='Extracted pdf')
axs[1].set_ylim(ax1.get_ylim())
plt.legend()
plt.show()

'''