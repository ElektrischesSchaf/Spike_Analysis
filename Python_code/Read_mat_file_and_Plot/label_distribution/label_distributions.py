import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

CWD_origin=os.getcwd()

label_hist_figure = os.path.join(CWD_origin, 'label_hist_figure')
if not os.path.exists(label_hist_figure):
    os.mkdir(label_hist_figure)

my_width = 16
my_height = 9
my_fontsize = 30

x_position_label_training = pd.read_csv('x_position_label_training.csv', dtype=float)    
x_position_label_training = np.array(x_position_label_training)

x_position_label_testing = pd.read_csv('x_position_label_testing.csv', dtype=float)    
x_position_label_testing = np.array(x_position_label_testing )

y_position_label_training = pd.read_csv('y_position_label_training.csv', dtype=float)    
y_position_label_training = np.array(y_position_label_training)

y_position_label_testing = pd.read_csv('y_position_label_testing.csv', dtype=float)    
y_position_label_testing = np.array(y_position_label_testing)

x_velocity_label_training = pd.read_csv('x_velocity_label_training.csv', dtype=float)    
x_velocity_label_training = np.array(x_velocity_label_training)

x_velocity_label_testing = pd.read_csv('x_velocity_label_testing.csv', dtype=float)    
x_velocity_label_testing = np.array(x_velocity_label_testing)

y_velocity_label_training = pd.read_csv('y_velocity_label_training.csv', dtype=float)    
y_velocity_label_training = np.array(y_velocity_label_training)

y_velocity_label_testing = pd.read_csv('y_velocity_label_testing.csv', dtype=float)    
y_velocity_label_testing = np.array(y_velocity_label_testing)


x_acceleration_label_training = pd.read_csv('x_acceleration_label_training.csv', dtype=float)    
x_acceleration_label_training = np.array(x_acceleration_label_training)

x_acceleration_label_testing = pd.read_csv('x_acceleration_label_testing.csv', dtype=float)    
x_acceleration_label_testing = np.array(x_acceleration_label_testing)

y_acceleration_label_training = pd.read_csv('y_acceleration_label_training.csv', dtype=float)    
y_acceleration_label_training = np.array(y_acceleration_label_training)

y_acceleration_label_testing = pd.read_csv('y_acceleration_label_testing.csv', dtype=float)    
y_acceleration_label_testing = np.array(y_acceleration_label_testing )

print(x_position_label_training.shape)


sns.set(font_scale=3)
sns.set_style("white")
plt.rcParams["figure.figsize"] = (my_width, my_height )
fig, axes = plt.subplots( 3 ,gridspec_kw={'height_ratios': [1,1,1],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05} , constrained_layout=True)

bins = [i for i in np.arange( -200, 200,  10)]

axes[0].hist(x_position_label_training[:,0], bins=bins, color='blue', alpha=0.6, label='x-axis training')
axes[0].hist(y_position_label_training[:,0], bins=bins, color='green', alpha=0.6,label='y-axis training')

axes[0].hist(x_position_label_testing[:,0], bins=bins, color='blue', alpha=0.3, label='x-axis testing')
axes[0].hist(y_position_label_testing[:,0], bins=bins, color='green', alpha=0.3, label='y-axis testing')

axes[0].set_xlabel('Position (mm)', fontsize=my_fontsize)
axes[0].set_yticks([])
axes[0].set_xlim([-150,150])
# axes[0].legend(loc="upper left", frameon=True, fontsize=my_fontsize)

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_visible(True)
axes[0].spines['left'].set_visible(False)


axes[1].hist(x_velocity_label_training[:,0], bins='auto', color='blue', alpha=0.6, label='x-axis training')
axes[1].hist(y_velocity_label_training[:,0], bins='auto', color='green', alpha=0.6,label='y-axis training')


axes[1].hist(x_velocity_label_testing[:,0], bins='auto', color='blue', alpha=0.3, label='x-axis testing')
axes[1].hist(y_velocity_label_testing[:,0], bins='auto', color='green', alpha=0.3, label='y-axis testing')

axes[1].set_xlabel('Velocity (mm/s)', fontsize=my_fontsize)
axes[1].set_yticks([])
axes[1].set_xlim([-200,200])
axes[1].legend(loc="upper right", frameon=True, fontsize=my_fontsize*0.7 , bbox_to_anchor=( 1.1 , 1.07 ) )

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_visible(True)
axes[1].spines['left'].set_visible(False)


bins = [i for i in np.arange( -2000, 2000,  50)]


axes[2].hist(x_acceleration_label_training[:,0], bins=bins, color='blue', alpha=0.6, label='x-axis training')
axes[2].hist(y_acceleration_label_training[:,0], bins=bins, color='green', alpha=0.6, label='y-axis training')

axes[2].hist(x_acceleration_label_testing[:,0], bins=bins, color='blue', alpha=0.3, label='x-axis testing')
axes[2].hist(y_acceleration_label_testing[:,0], bins=bins, color='green', alpha=0.3, label='y-axis testing')

axes[2].set_xlabel('Acceleration (mm/$\mathrm{s}^{\mathrm{2}}$)', fontsize=my_fontsize)
axes[2].set_yticks([])
axes[2].set_xlim([-2000,2000])
# axes[2].legend(loc="upper left", frameon=True, fontsize=my_fontsize)

axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].spines['bottom'].set_visible(True)
axes[2].spines['left'].set_visible(False)

plt.savefig( label_hist_figure+'/'+ 'all_label_distribution.png')

# plt.show()


plt.cla()
plt.clf()
fig.clear()



