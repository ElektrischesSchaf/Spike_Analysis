import numpy as np
import pandas as pd
import h5py
import os
import numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)

CWD = os.getcwd()
plot_path = os.path.join(CWD, 'My_plot_bidir_vs_makin')
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

my_model_name = 'atten_LN_bidir'
makin_model_name = 'rEFH_dynamic'
my_fontsize=35

x_pos = pd.read_csv( my_model_name+'/pos.csv', usecols=["x-axis"] , dtype=float)
x_pos = x_pos.values
print(x_pos.shape)

y_pos = pd.read_csv( my_model_name+'/pos.csv', usecols=["y-axis"] , dtype=float)
y_pos = y_pos.values

x_vel = pd.read_csv( my_model_name+'/vel.csv', usecols=["x-axis"] , dtype=float)
x_vel = x_vel.values

y_vel = pd.read_csv( my_model_name+'/vel.csv', usecols=["y-axis"] , dtype=float)
y_vel = y_vel.values

x_acc = pd.read_csv( my_model_name+'/acc.csv', usecols=["x-axis"] , dtype=float)
x_acc = x_acc.values

y_acc = pd.read_csv( my_model_name+'/acc.csv', usecols=["y-axis"] , dtype=float)
y_acc = y_acc.values

rEFH_x_pos = pd.read_csv( makin_model_name + '/all_SNR.csv', usecols=["x-pos"] , dtype=float)
rEFH_x_pos = rEFH_x_pos.values

rEFH_y_pos = pd.read_csv( makin_model_name + '/all_SNR.csv', usecols=["y-pos"] , dtype=float)
rEFH_y_pos = rEFH_y_pos.values

rEFH_x_vel = pd.read_csv( makin_model_name + '/all_SNR.csv', usecols=["x-vel"] , dtype=float)
rEFH_x_vel = rEFH_x_vel.values

rEFH_y_vel = pd.read_csv( makin_model_name + '/all_SNR.csv', usecols=["y-vel"] , dtype=float)
rEFH_y_vel = rEFH_y_vel.values

rEFH_x_acc = pd.read_csv( makin_model_name + '/all_SNR.csv', usecols=["x-acc"] , dtype=float)
rEFH_x_acc = rEFH_x_acc.values

rEFH_y_acc = pd.read_csv( makin_model_name + '/all_SNR.csv', usecols=["y-acc"] , dtype=float)
rEFH_y_acc = rEFH_y_acc.values


def drawing( rEFH_dynamic_result, my_model_result, axis_type, kinematic_type ):
    plt.figure(figsize=(12,12))
    plt.scatter( rEFH_dynamic_result[:37], my_model_result[:37], s=100, color='blue', label='Indy' )
    plt.scatter( rEFH_dynamic_result[37:], my_model_result[37:], s=100, color='green' , label='Loco ')

    plt.plot([0,10],[0,10], color='black')
    plt.xlim([0,10])
    plt.ylim([0,10])

    if makin_model_name == 'rEFH_dynamic':
        plt.xlabel('SNR(dB), rEFH(+KF)',fontsize=my_fontsize)
    if my_model_name == 'atten_LN_bidir':
        plt.ylabel('SNR(dB), Bidir GRU (+LN & Atten.)',fontsize=my_fontsize)

    plt.xticks(fontsize=my_fontsize)
    plt.yticks(fontsize=my_fontsize)
    plt.legend(fontsize=my_fontsize, loc='lower right')

    model.fit( rEFH_dynamic_result.reshape(-1, 1).astype(np.float32), my_model_result.reshape(-1, 1).astype(np.float32) )
    predict = model.predict(rEFH_dynamic_result)
    plt.plot( rEFH_dynamic_result, predict , color='red', linestyle='solid' )

    # plt.show()
    yee_1 = '0'
    yee_2 = '0'

    title_text_1 = '0'
    title_text_2 = '0'

    if axis_type=='x':
        yee_1 = 'x_'
        title_text_2 = ', x-axis'
    if axis_type=='y':
        yee_1 = 'y_'
        title_text_2 = ', y-axis'
    if kinematic_type == 'pos':
        yee_2 = 'pos'        
        title_text_1 = 'Position'        
    if kinematic_type == 'vel':
        yee_2 = 'vel'        
        title_text_1 = 'Velocity'
    if kinematic_type == 'acc':
        yee_2 = 'acc'        
        title_text_1 = 'Acceleration'

    plt.title( title_text_1 + title_text_2 , fontsize=my_fontsize)
    plt.tight_layout()
    plt.savefig( plot_path + '/' + yee_1 + yee_2 +'.png' )


drawing(rEFH_x_pos, x_pos, 'x', 'pos')
drawing(rEFH_y_pos, y_pos, 'y', 'pos')
drawing(rEFH_x_vel, x_vel, 'x', 'vel')
drawing(rEFH_y_vel, y_vel, 'y', 'vel')
drawing(rEFH_x_acc, x_acc, 'x', 'acc')
drawing(rEFH_y_acc, y_acc, 'y', 'acc')