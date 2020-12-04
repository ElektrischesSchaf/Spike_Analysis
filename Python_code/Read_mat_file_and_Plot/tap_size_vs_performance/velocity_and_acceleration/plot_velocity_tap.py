import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import numpy as np

my_fontsize=45


plt.figure(figsize=(16,9))
FILE_PATH = './tap_sizes_results/'
ALL_List_FILE = os.listdir(FILE_PATH)
ALL_List_FILE.sort()
List_FILE=ALL_List_FILE[:]
file_list=List_FILE

the_session_name = 'indy_20161007_02'
kinematci_types = 'x_and_y_vel'

x_score_all_list = []
y_score_all_list = []

tap_sizes_all = []

for file_num in file_list:
    if file_num.startswith("GRU"):
        file_name = FILE_PATH+ '/' + file_num + '/' + kinematci_types +'/'+ the_session_name +'/' + 'csv_files' +'/' +'R_square_this_session' + '.csv'

        yee=file_num.split('_')
        tap_sizes = int(yee[-1])

        df = pd.read_csv(file_name)
        x_axis_score = df['x-axis'].to_numpy()
        y_axis_score = df['y-axis'].to_numpy()
        
        x_score_all_list.append(x_axis_score)
        y_score_all_list.append(y_axis_score)
        tap_sizes_all.append(tap_sizes)

plt.figure(figsize=(16,9))

if kinematci_types == 'x_and_y_pos':
    x_label='x-position'
    y_label='y-position'
if kinematci_types == 'x_and_y_vel':
    x_label='x-velocity'
    y_label='y-velocity'
if kinematci_types == 'x_and_y_acc':
    x_label='x-acceleration'
    y_label='y-acceleration'

plt.plot(tap_sizes_all, x_score_all_list, linewidth=5, label=x_label, color='blue')
plt.plot(tap_sizes_all, y_score_all_list, linewidth=5, label=y_label, color='green')

x_regression_a = np.polyfit(tap_sizes_all, x_score_all_list, 3)
y_regression_a = np.polyfit(tap_sizes_all, y_score_all_list, 3)

x_regression_a=np.squeeze(x_regression_a)
y_regression_a=np.squeeze(y_regression_a)

x_regression_p = np.poly1d(x_regression_a)
y_regression_p = np.poly1d(y_regression_a)

plt.plot(tap_sizes_all, x_regression_p(tap_sizes_all), linewidth=3, color='blue', alpha=0.7)
plt.plot(tap_sizes_all, y_regression_p(tap_sizes_all), linewidth=3, color='green', alpha=0.7 )

plt.xlabel('Tap sizes' , fontsize=my_fontsize)
plt.xticks(fontsize=my_fontsize*0.8)
plt.ylabel('$\mathrm{R}^{\mathrm{2}}$', fontsize=my_fontsize )
plt.yticks(fontsize=my_fontsize*0.8)
plt.legend(loc='lower right', fontsize=my_fontsize*0.8 )
plt.xlim([tap_sizes_all[0], tap_sizes_all[-1]])

plt.tight_layout()
if kinematci_types == 'x_and_y_pos':
    plt.savefig('../tap_sizes_position.png')
if kinematci_types == 'x_and_y_vel':
    plt.savefig('../tap_sizes_velocity.png')
if kinematci_types == 'x_and_y_acc':
    plt.savefig('../tap_sizes_acceleration.png')

# plt.show()