import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

CWD_origin=os.getcwd()

label_hist_figure = os.path.join(CWD_origin, 'label_hist_figure')
if not os.path.exists(label_hist_figure):
    os.mkdir(label_hist_figure)

my_width=16
my_height=3
my_fontsize=15

x_position_label_training = pd.read_csv('x_position_label_training.csv', dtype=float)    
x_position_label_training = np.array(x_position_label_training)

y_position_label_training = pd.read_csv('y_position_label_training.csv', dtype=float)    
y_position_label_training = np.array(y_position_label_training)

x_velocity_label_training = pd.read_csv('x_velocity_label_training.csv', dtype=float)    
x_velocity_label_training = np.array(x_velocity_label_training)

y_velocity_label_training = pd.read_csv('y_velocity_label_training.csv', dtype=float)    
y_velocity_label_training = np.array(y_velocity_label_training)

x_acceleration_label_training = pd.read_csv('x_acceleration_label_training.csv', dtype=float)    
x_acceleration_label_training = np.array(x_acceleration_label_training)

y_acceleration_label_training = pd.read_csv('y_acceleration_label_training.csv', dtype=float)    
y_acceleration_label_training = np.array(y_acceleration_label_training)

print(x_position_label_training.shape)


fig=plt.figure(figsize=(my_width,my_height))
plt.rcParams['xtick.labelsize'] = my_fontsize
plt.rcParams['ytick.labelsize'] = my_fontsize

bins = [i for i in np.arange( -200, 200,  10)]

plt.hist(x_position_label_training[:,0], bins=bins, color='blue', alpha=0.5, label='x')
plt.hist(y_position_label_training[:,0], bins=bins, color='green', alpha=0.5,label='y')

plt.xlabel('Position (mm)', fontsize=my_fontsize)
plt.yticks([])
plt.xlim([-150,150])
plt.legend(loc="upper left", frameon=True, fontsize=my_fontsize)
plt.tight_layout()

plt.savefig( label_hist_figure+'/'+ 'position.png')
# plt.show()

plt.cla()
plt.clf()
fig.clear()

fig=plt.figure(figsize=(my_width,my_height))
plt.rcParams['xtick.labelsize'] = my_fontsize
plt.rcParams['ytick.labelsize'] = my_fontsize
plt.hist(x_velocity_label_training[:,0], bins='auto', color='blue', alpha=0.5, label='x')
plt.hist(y_velocity_label_training[:,0], bins='auto', color='green', alpha=0.5,label='y')

plt.xlabel('Velocity (mm/s)', fontsize=my_fontsize)
plt.yticks([])
plt.xlim([-200,200])
plt.legend(loc="upper left", frameon=True, fontsize=my_fontsize)
plt.tight_layout()

plt.savefig( label_hist_figure+'/'+ 'velocity.png')
# plt.show()

plt.cla()
plt.clf()
fig.clear()

bins = [i for i in np.arange( -2000, 2000,  50)]

fig=plt.figure(figsize=(my_width,my_height))
plt.rcParams['xtick.labelsize'] = my_fontsize
plt.rcParams['ytick.labelsize'] = my_fontsize
plt.hist(x_acceleration_label_training[:,0], bins=bins, color='blue', alpha=0.5, label='x')
plt.hist(y_acceleration_label_training[:,0], bins=bins, color='green', alpha=0.5,label='y')

plt.xlabel('Acceleration (mm/$s^2$)', fontsize=my_fontsize)
plt.yticks([])
plt.xlim([-2000,2000])
plt.legend(loc="upper left", frameon=True, fontsize=my_fontsize)
plt.tight_layout()

plt.savefig( label_hist_figure+'/'+ 'acceleration.png')


plt.cla()
plt.clf()
fig.clear()
