import numpy as np
import pandas as pd
import h5py
import os
import numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)


CWD = os.getcwd()

plot_path = os.path.join(CWD, 'My_plot')
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

# vs Makin et al. 2018
plot_path_1 = os.path.join(plot_path, 'My_plot_bidir_vs_makin')
if not os.path.exists(plot_path_1):
    os.mkdir(plot_path_1)

# one-way / bidir
plot_path_2 = os.path.join(plot_path, 'My_plot_bidir_vs_one_way')
if not os.path.exists(plot_path_2):
    os.mkdir(plot_path_2)

# special
plot_path_3 = os.path.join(plot_path, 'bir_attention_with_LN_inside_vs_no_LN_inside')
if not os.path.exists(plot_path_3):
    os.mkdir(plot_path_3)

# w/o attention
plot_path_4 = os.path.join(plot_path, 'My_plot_bidir_vs_no_atten')
if not os.path.exists(plot_path_4):
    os.mkdir(plot_path_4)

# w/o LN
plot_path_5 = os.path.join(plot_path, 'My_plot_bidir_vs_no_LN')
if not os.path.exists(plot_path_5):
    os.mkdir(plot_path_5)

my_model_name_for_y = 'atten_LN_bidir' # atten_LN_bidir
makin_model_name = 'rEFH_dynamic'

my_model_name_for_x = 'bidir_atten_no_LN' # atten_LN_oneway , atten_LN_bidir_no_LN_inside , bidir_LN_no_atten , bidir_atten_no_LN
plot_path_for_this = plot_path_5

my_fontsize=35

def read_my_result_1_for_y_scatter( my_model_name ):
    my_model_name = my_model_name
    x_pos = pd.read_csv( my_model_name+'/pos.csv', usecols=["x-axis"] , dtype=float)
    x_pos = x_pos.values

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
    return x_pos, y_pos, x_vel, y_vel, x_acc, y_acc

# def read_my_result_2_for_x_scatter( my_model_name_for_x ):
#     my_model_name = my_model_name_for_x
#     x_pos = pd.read_csv( my_model_name+'/pos.csv', usecols=["x-axis"] , dtype=float)
#     x_pos = x_pos.values

#     y_pos = pd.read_csv( my_model_name+'/pos.csv', usecols=["y-axis"] , dtype=float)
#     y_pos = y_pos.values

#     x_vel = pd.read_csv( my_model_name+'/vel.csv', usecols=["x-axis"] , dtype=float)
#     x_vel = x_vel.values

#     y_vel = pd.read_csv( my_model_name+'/vel.csv', usecols=["y-axis"] , dtype=float)
#     y_vel = y_vel.values

#     x_acc = pd.read_csv( my_model_name+'/acc.csv', usecols=["x-axis"] , dtype=float)
#     x_acc = x_acc.values

#     y_acc = pd.read_csv( my_model_name+'/acc.csv', usecols=["y-axis"] , dtype=float)
#     y_acc = y_acc.values
#     return x_pos, y_pos, x_vel, y_vel, x_acc, y_acc

def read_makin_rEFH_result(makin_model_name):
    makin_model_name=makin_model_name
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

    return rEFH_x_pos, rEFH_y_pos, rEFH_x_vel, rEFH_y_vel, rEFH_x_acc, rEFH_y_acc

def drawing_with_makin(plot_path, rEFH_dynamic_result, my_model_result, axis_type, kinematic_type , my_model_name_for_y):
    plot_path=plot_path
    my_model_name_for_y=my_model_name_for_y

    plt.figure(figsize=(12,12))
    plt.scatter( rEFH_dynamic_result[:37], my_model_result[:37], s=100, color='blue', label='Indy' )
    plt.scatter( rEFH_dynamic_result[37:], my_model_result[37:], s=100, color='green' , label='Loco ')

    plt.plot([0,10],[0,10], color='black')
    plt.xlim([0,10])
    plt.ylim([0,10])

    if makin_model_name == 'rEFH_dynamic':
        plt.xlabel('SNR(dB), rEFH(+KF)',fontsize=my_fontsize)
    if my_model_name_for_y == 'atten_LN_bidir':
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

def drawing_with_my_two_results( plot_path_2, my_model_result_x, my_model_result_y, axis_type, kinematic_type , my_model_name_for_x,  my_model_name_for_y):
    plot_path=plot_path_2
    my_model_name_for_x = my_model_name_for_x
    my_model_name_for_y = my_model_name_for_y
    plt.figure(figsize=(12,12))
    plt.scatter( my_model_result_x[:37], my_model_result_y[:37], s=100, color='blue', label='Indy' )
    plt.scatter( my_model_result_x[37:], my_model_result_y[37:], s=100, color='green' , label='Loco ')

    plt.plot([0,10],[0,10], color='black')
    plt.xlim([0,10])
    plt.ylim([0,10])

    if my_model_name_for_x == 'atten_LN_oneway': # path 2
        plt.xlabel('SNR(dB), One-way GRU (+LN & Atten.)',fontsize=my_fontsize)
    if my_model_name_for_x =='atten_LN_bidir_no_LN_inside': # special path 3
        plt.xlabel('SNR(dB), Bidir GRU (+LN & Atten. no LN )',fontsize=my_fontsize)
    if my_model_name_for_x =='bidri_LN_no_atten': # path 4
        plt.xlabel('SNR(dB), Bidir GRU (+ LN )',fontsize=my_fontsize)
    if my_model_name_for_x == 'bidir_atten_no_LN': # path 5
        plt.xlabel('SNR(dB), Bidir GRU (+ Atten.)',fontsize=my_fontsize)

    if my_model_name_for_y == 'atten_LN_bidir':
        plt.ylabel('SNR(dB), Bidir GRU (+LN & Atten.)',fontsize=my_fontsize)

    plt.xticks(fontsize=my_fontsize)
    plt.yticks(fontsize=my_fontsize)
    plt.legend(fontsize=my_fontsize, loc='lower right')

    model.fit( my_model_result_x.reshape(-1, 1).astype(np.float32), my_model_result_y.reshape(-1, 1).astype(np.float32) )
    predict = model.predict(my_model_result_x)
    plt.plot( my_model_result_x, predict , color='red', linestyle='solid' )

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



x_pos_1, y_pos_1, x_vel_1, y_vel_1, x_acc_1, y_acc_1 = read_my_result_1_for_y_scatter (my_model_name_for_y )

rEFH_x_pos, rEFH_y_pos, rEFH_x_vel, rEFH_y_vel, rEFH_x_acc, rEFH_y_acc = read_makin_rEFH_result(makin_model_name)

x_pos_2, y_pos_2, x_vel_2, y_vel_2, x_acc_2, y_acc_2 = read_my_result_1_for_y_scatter( my_model_name_for_x )



drawing_with_makin(plot_path_1, rEFH_x_pos, x_pos_1, 'x', 'pos',my_model_name_for_y)
drawing_with_makin(plot_path_1, rEFH_y_pos, y_pos_1, 'y', 'pos',my_model_name_for_y)
drawing_with_makin(plot_path_1, rEFH_x_vel, x_vel_1, 'x', 'vel',my_model_name_for_y)
drawing_with_makin(plot_path_1, rEFH_y_vel, y_vel_1, 'y', 'vel',my_model_name_for_y)
drawing_with_makin(plot_path_1, rEFH_x_acc, x_acc_1, 'x', 'acc',my_model_name_for_y)
drawing_with_makin(plot_path_1, rEFH_y_acc, y_acc_1, 'y', 'acc',my_model_name_for_y)



drawing_with_my_two_results(plot_path_for_this,  x_pos_2, x_pos_1, 'x', 'pos', my_model_name_for_x, my_model_name_for_y)
drawing_with_my_two_results(plot_path_for_this,  y_pos_2, y_pos_1, 'y', 'pos', my_model_name_for_x, my_model_name_for_y)
drawing_with_my_two_results(plot_path_for_this,  x_vel_2, x_vel_1, 'x', 'vel', my_model_name_for_x, my_model_name_for_y)
drawing_with_my_two_results(plot_path_for_this,  y_vel_2, y_vel_1, 'y', 'vel', my_model_name_for_x, my_model_name_for_y)
drawing_with_my_two_results(plot_path_for_this,  x_acc_2, x_acc_1, 'x', 'acc', my_model_name_for_x, my_model_name_for_y)
drawing_with_my_two_results(plot_path_for_this,  y_acc_2, y_acc_1, 'y', 'acc', my_model_name_for_x, my_model_name_for_y)