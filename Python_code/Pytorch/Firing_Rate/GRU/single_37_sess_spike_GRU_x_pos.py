# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
# Figures
import imageio
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
width_two=0.2
# Data Processing
import pandas as pd
import json
import math
import numpy as np
import h5py
from tqdm import tqdm_notebook as tqdm
from tqdm import trange
from sklearn import datasets, svm, metrics
# Regression Problem Evaluation Methods
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
# Read/Write file
import os
CWD_origin=os.getcwd()
import shutil

import time
tStart=time.time()

# My module
import sys
sys.path.append("../../..") # Adds higher directory to python modules path.
import data_processing.parameters as my_parameters
import data_processing.load_mat_file as load_mat_file
my_parameters=my_parameters.my_parameters()
mat_file_processing=load_mat_file.mat_file_processing()

# Deep leaning module
# from  Deep_Learning_Models.GRU_one_stream import GRUModel
from  Deep_Learning_Models.GRU_one_stream_self_Atten import GRUModel
from Deep_Learning_Models.Abstract_Dataset_Class import AbstractDataset

# Make file list
kinematic_variable_type='x_pos' # x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc
FILE_PATH = '../../../../Dataset/Sorted_Spike_Dataset/'
ALL_List_FILE = os.listdir(FILE_PATH)
ALL_List_FILE.sort()
List_FILE=ALL_List_FILE[:]
session_file_list=List_FILE

# Neural Network Hyperparameters
model_name='GRU_with_Spike_Single_37_Session'
MAX_EPOCH= 50
LEARNING_RATE = 1e-5
NUMBER_OF_LAYERS = 2
BATCH_SIZE = 16
HIDDEN_DIMENSION = 100
max_timestep = 20
# Model Performance Lists
R_square_across_all_sessions=[]
SNR_across_all_sessions=[]
RMSE_across_all_sessions=[]
best_epoch_arcoss_all_sessions=[]
person_correlation_coefficient_across_all_sessions=[]

# session control start
for session_k in range(len(session_file_list)):

    session_name=str(session_file_list[session_k])[:-4]
    file_name_1='../../../../Dataset/Sorted_Spike_Dataset/'+ session_name +'.mat'
    # file_list=[file_name_1, file_name_2, file_name_3, file_name_4, file_name_5, file_name_6]

    time_stamp_64ms=[]

    # Auto-assigned parameters
    testing_data_index=0
    channel_number=0
    units_have_value=0

    # Parameters should be assigned
    the_sampling_rate=my_parameters.the_sampling_rate
    file_numbers=my_parameters.file_numbers
    time_lag=my_parameters.time_lag

    with_sorted_spikes=True
    include_hash_unit=my_parameters.include_hash_unit

    print('In session '+ session_name + ': ' + '\n' )

    # Load Spike Firing Rate
    [firing_rate_cell, channel_number, testing_data_index, time_stamp_64ms, unit_number]=mat_file_processing.get_spike_bins_matrix(file_name_1, the_sampling_rate, time_stamp_64ms, include_hash_unit)

    # Get channel and unit numbers
    channel_numbers_in_this_dataset=96
    units_numbers_in_this_dataset=unit_number

    if with_sorted_spikes==True:
        feature_numbers=channel_numbers_in_this_dataset*units_numbers_in_this_dataset
    else:
        feature_numbers=channel_numbers_in_this_dataset

    # Create empty arrrays from data
    [X_for_training, X_for_prediction, 
    x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
    x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
    x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing]=mat_file_processing.create_empty_traing_and_testing_label(feature_numbers)

    [time_stamp_64ms, x_position_label, y_position_label, z_position_label, x_velocity_label, y_velocity_label, z_velocity_label, x_acceleration_label, y_acceleration_label,  z_acceleration_label]=mat_file_processing.get_labels(file_name_1, the_sampling_rate, time_stamp_64ms)

    # Extract firing_rate_cell with rows have length bigger than zero
    firing_rate_final=[] # not[[]]
    for row_index in range( len( firing_rate_cell) ):   
        if len(firing_rate_cell[row_index]):
            firing_rate_final.append( firing_rate_cell[row_index] )
            units_have_value+=1

    '''
    for row_index in range( len( firing_rate_final) ):            
        print('length of firing_rate_final['+ str(row_index) +']: ',end='')
        print(len(firing_rate_final[row_index]))
    '''

    print('\n')

    firing_rate_matrix=np.array(firing_rate_final)
    print('firing_rate_matrix shape: ', firing_rate_matrix.shape) #  in indy_20160407_02 (226, 12777) eliminated null units, (288, 12777) with all 96X3 units
    print('\n')


    # New Without spike sorting:
    if with_sorted_spikes==False:
        with_sorting_firing_rate=firing_rate_matrix.copy()
        firing_rate_matrix=np.zeros([ channel_number, firing_rate_matrix.shape[1] ])
        print('firing_rate_matrix shape: ', firing_rate_matrix.shape)  # (96, 12777)
        print('with_sorting_firing_rate shape: ', with_sorting_firing_rate.shape) # (288, 12777)
        print('\n')

        for i in range(with_sorting_firing_rate.shape[1]):
            index=0
            k=0
            while index < channel_number:
                
                all_units_firing_rate_sum=0
                for unit_index in range( int(with_sorting_firing_rate.shape[0] / channel_number) ):
                    all_units_firing_rate_sum+=with_sorting_firing_rate[k+unit_index][i]
                firing_rate_matrix[index][i]=all_units_firing_rate_sum

                # firing_rate_matrix[index][i]=with_sorting_firing_rate[k][i]+with_sorting_firing_rate[k+1][i]+with_sorting_firing_rate[k+2][i]

                index = index + 1
                k = k+ unit_number
        print('firing_rate_matrix shape: ', firing_rate_matrix.shape)  # (96, 12777)
        print('with_sorting_firing_rate shape: ', with_sorting_firing_rate.shape) # (288, 12777)
        print('\n')

    firing_rate_matrix=np.transpose(firing_rate_matrix)        
    feature_numbers_of_firing_rate = firing_rate_matrix.shape[1]

    X=firing_rate_matrix.astype(np.float32)
    print('features list shape: ',end='')
    print( X.shape ) # X is the feature matrix,  (12777, 288) in indy_20160407_02
    print('\n')

    # Cross Session Data Concatenation
    [X_for_training, X_for_prediction,
    x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
    x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
    x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing] = mat_file_processing.cross_session_data_concatenation(
    feature_numbers_of_firing_rate, X, testing_data_index, X_for_training, X_for_prediction,
    x_position_label, y_position_label, z_position_label, x_velocity_label, y_velocity_label, z_velocity_label, x_acceleration_label, y_acceleration_label, z_acceleration_label,
    x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
    x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
    x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing)

    print('shape of X_for_training before', X_for_training.shape)


    # Normalize Firing rate
    # the_mean=np.mean(X_for_training)
    # the_std=np.std(X_for_training)
    # X_for_training = (X_for_training - the_mean )/ the_std
    # X_for_prediction = (X_for_prediction - the_mean )/ the_std

    # Processing max orders
    order_num=max_timestep-1
    [X_for_training, X_for_prediction,
    x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
    x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
    x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing] = mat_file_processing.max_order_preparation(
    order_num, feature_numbers, X_for_training, X_for_prediction,
    x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
    x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
    x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing)
    
    print('shape of X_for_training after', X_for_training.shape)
    print('shape of x_position_label_training after', x_position_label_training.shape)

    print('\nshape of X_for_prediction after', X_for_prediction.shape)
    print('shape of x_position_label_testing after', x_position_label_testing.shape)

    # Write features and label from each session to csv files
    CWD = CWD_origin
    if model_name not in CWD:
        CWD = os.path.join(CWD, model_name)
        if not os.path.exists(CWD):
            os.mkdir(CWD)

    CWD=os.path.join(CWD, kinematic_variable_type)
    if not os.path.exists(CWD):
        os.mkdir(CWD)
    
    bar_plot_path=os.path.join(CWD, 'bar_plot_across_sessions')
    if not os.path.exists(bar_plot_path):
        os.mkdir(bar_plot_path)

    if session_name not in CWD:
        CWD = os.path.join(CWD, session_name)
        if not os.path.exists(CWD):
            os.mkdir(CWD)

    save_epoch_path=os.path.join(CWD,'save')
    if not os.path.exists(save_epoch_path):
        os.makedirs(save_epoch_path)

    csv_path=os.path.join(CWD,'csv_files')
    if not os.path.exists(csv_path):
        os.mkdir(str(csv_path))

    plot_path = os.path.join(CWD, 'plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)


    df = pd.DataFrame(X_for_training)
    df.to_csv(os.path.join(csv_path, 'trainset_feature_matrix.csv'), index=False)

    df = pd.DataFrame(X_for_prediction)
    df.to_csv(os.path.join(csv_path, 'testset_feature_matrix.csv'), index=False)
    
    if kinematic_variable_type=='x_pos':
        df=pd.DataFrame(x_position_label_training)
        df.to_csv(os.path.join(csv_path,'x_position_label_training.csv'), index=False)

        df=pd.DataFrame(x_position_label_testing)
        df.to_csv(os.path.join(csv_path,'x_position_label_testing.csv'), index=False)

    if kinematic_variable_type=='y_pos':
        df=pd.DataFrame(y_position_label_training)
        df.to_csv(os.path.join(csv_path,'y_position_label_training.csv'), index=False)

        df=pd.DataFrame(y_position_label_testing)
        df.to_csv(os.path.join(csv_path,'y_position_label_testing.csv'), index=False)

    # read from csv file
    training_x=pd.read_csv(os.path.join(csv_path, 'trainset_feature_matrix.csv'), dtype=float)
    training_x = torch.from_numpy(training_x.values) # .values can turn pandas dataframe to numpy array
    training_x=training_x.float()

    testing_x=pd.read_csv(os.path.join(csv_path, 'testset_feature_matrix.csv'), dtype=float)
    testing_x = torch.from_numpy(testing_x.values) # .values can turn pandas dataframe to numpy array
    testing_x=testing_x.float()

    if kinematic_variable_type=='x_pos':
        training_y=pd.read_csv(os.path.join(csv_path,'x_position_label_training.csv'), dtype=float)    
        training_y = torch.from_numpy(training_y.values)    
        training_y=training_y.float()

        testing_y=pd.read_csv(os.path.join(csv_path,'x_position_label_testing.csv'), dtype=float)    
        testing_y = torch.from_numpy(testing_y.values)    
        testing_y=testing_y.float()

    if kinematic_variable_type=='y_pos':
        training_y=pd.read_csv(os.path.join(csv_path,'y_position_label_training.csv'), dtype=float)    
        training_y = torch.from_numpy(training_y.values)    
        training_y=training_y.float()

        testing_y=pd.read_csv(os.path.join(csv_path,'y_position_label_testing.csv'), dtype=float)    
        testing_y = torch.from_numpy(testing_y.values)    
        testing_y=testing_y.float()

    # Normalize label
    # training_y_mean=torch.mean(training_y)
    # training_y_std=torch.std(training_y)

    # General Neural Network Hyperparameters
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    max_epoch=MAX_EPOCH

    # GRU Hyperparameters
    hidden_dim = HIDDEN_DIMENSION
    layer_dim = NUMBER_OF_LAYERS
    output_dim = 1

    # Training / Testing AbstractDataset
    training_dataset=AbstractDataset(training_x, training_y)
    testing_dataset=AbstractDataset(testing_x, testing_y)

    # TODO collate_fn
    # train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=False, collate_fn=training_dataset.collate_fn)
    # train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=False)
    # test_loader=torch.utils.data.DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=False)    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = GRUModel(input_dim=feature_numbers, hidden_dim=hidden_dim, max_timestep=max_timestep, layer_dim=layer_dim, output_dim=output_dim)     # define the network
    # print(net)  # net architecture
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    net.to(device)
    history = {'train':[],'test':[]}

    def _run_epoch(epoch, mode):
        net.train(True)
        if mode=='train':
            descrpition='Train'
            dataset=training_dataset
            schuffle=False
        else:
            descrpition='Test'
            dataset=testing_dataset
            shuffle=False
        dataloader=torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=False
                                                #collate_fn=dataset.collate_fn,
                                                )
        trange=tqdm(enumerate(dataloader), total=len(dataloader), desc=descrpition)
        loss=0

        my_prediction = []
        real_y_all=[]

        for i, (x, y) in trange:

            # GRU batch
            # if(x.size()[0] is not batch_size):
            #     continue

            o_labels, batch_loss = _run_iter(x,y)

            if mode=='train':
                optimizer.zero_grad()   # clear gradients for next train
                batch_loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients

            loss+=batch_loss.item()

            real_y=y.cpu().data.numpy()
            for ele in o_labels.cpu().data.numpy():
                my_prediction.append(ele)

            for ele in real_y:
                real_y_all.append(ele)
            
            R_square=r2_score( real_y_all, my_prediction)

            trange.set_postfix(loss=loss/(i+1), R_square=R_square)

        if mode=='train':
            history['train'].append({'loss':loss/len(trange), 'R^2': R_square })
            # writer.add_scalar('Loss/train', loss/len(trange), epoch)
        else:
            history['test'].append({'loss':loss/len(trange), 'R^2': R_square })
            print('R-square: ', R_square, '\n')
            # writer.add_scalar('Loss/test', loss/len(trange), epoch)
        trange.close()

    def _run_iter(x,y):
        feature = x.to(device)
        labels = y.to(device)

        o_labels = net(feature)
        # o_labels=o_labels*training_y_std+training_y_mean

        l_loss = loss_func(o_labels, labels)
        return o_labels, l_loss

    def save(epoch):
        torch.save(net.state_dict(), os.path.join( save_epoch_path, 'model.pkl.'+str(epoch) ))
        with open( os.path.join( save_epoch_path, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)

    for epoch in range(max_epoch):
        print('Epoch: {}'.format(epoch))
        _run_epoch(epoch, 'train')
        _run_epoch(epoch, 'test')
        save(epoch)

    # Plot the training results 
    with open(os.path.join(save_epoch_path, 'history.json'), 'r') as f:
        history = json.loads(f.read())
        
    train_loss = [l['loss'] for l in history['train']]
    valid_loss = [l['loss'] for l in history['test']]

    train_R_square = [l['R^2'] for l in history['train']]
    valid_R_square = [l['R^2'] for l in history['test']]


    plt.figure(figsize=(7,5))
    plt.title( 'Loss' )
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='test')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path + '/' + model_name+"_Loss.png")

    plt.figure(figsize=(7,5))
    plt.title( 'performance' )
    plt.plot(train_R_square, label='train')
    plt.plot(valid_R_square, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('R square')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path + '/' +model_name+"_R-square.png")

    best_score, best_epoch=max([[l['R^2'], idx] for idx, l in enumerate(history['test'])])
    print('best_score= ', best_score, ', best_epoch= ', best_epoch, '\n')    
    best_epoch_arcoss_all_sessions.append(best_epoch)
    print('Best R-square score ', max([[l['R^2'], idx] for idx, l in enumerate(history['test'])]))

    # Testing
    best_model=best_epoch # TODO
    net.load_state_dict(state_dict=torch.load(os.path.join(save_epoch_path, 'model.pkl.{}'.format(best_model))))
    net.train(False)
    # start testing
    dataloader = DataLoader(dataset=testing_dataset,
                                batch_size=batch_size,
                                shuffle=False
                                #collate_fn=testData.collate_fn,
                                #num_workers=8
                                )
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
    my_prediction = []
    real_y_all=[]

    for i, (x, testing_y) in trange:

        # GRU batch
        # if(x.size()[0] is not batch_size):
        #     continue

        o_labels = net(x.to(device))
        # o_labels=o_labels*training_y_std+training_y_mean

        real_y=testing_y.cpu().data.numpy()
        for ele in o_labels.cpu().data.numpy():
            my_prediction.append( float(ele) )

        for ele in real_y:
            real_y_all.append( float(ele) )
    shutil.rmtree(save_epoch_path)

    testing_data_r_square=r2_score( real_y_all, my_prediction)
    testing_data_SNR=-10*math.log10(1-testing_data_r_square)
    testing_data_RMSE=np.sqrt(mean_squared_error(real_y_all,my_prediction))
    PCC=pearsonr(real_y_all,my_prediction)
    Ground_Truth=real_y_all
    print('\n R-square: ', testing_data_r_square, ' RMSE: ', testing_data_RMSE, ', pearsonr=', PCC[0], '\n')

    R_square_across_all_sessions.append(testing_data_r_square)
    SNR_across_all_sessions.append(testing_data_SNR)
    RMSE_across_all_sessions.append(testing_data_RMSE)
    person_correlation_coefficient_across_all_sessions.append(PCC[0])

    # Plotting the kinematic variable reconstructure figure
    plt.figure(figsize=(32, 9))
    plotting_time_elapsed=time_stamp_64ms[testing_data_index:]
    plotting_time_elapsed=plotting_time_elapsed[max_timestep:]

    df = pd.DataFrame( plotting_time_elapsed )
    df.to_csv(os.path.join(csv_path, 'plotting_time_elapsed.csv'), index=False, header=False)

    df = pd.DataFrame( my_prediction )
    df.to_csv(os.path.join(csv_path, 'my_prediction.csv'), index=False, header=False)

    if kinematic_variable_type=='x_pos':
        plt.plot( plotting_time_elapsed, my_prediction, 'b', linewidth=5, label='Prediction' )
        plt.plot( plotting_time_elapsed, Ground_Truth, 'r', linewidth=5, label='Actual', alpha=0.7)
        plt.title( session_name + ' x-position prediction' , fontsize=30, color="black")

        df = pd.DataFrame( Ground_Truth )
        df.to_csv(os.path.join(csv_path, 'Ground_Truth_x_pos.csv'), index=False, header=False) 

    if kinematic_variable_type=='y_pos':
        plt.plot( plotting_time_elapsed, my_prediction, 'b', linewidth=5, label='Prediction' )
        plt.plot( plotting_time_elapsed, Ground_Truth, 'r', linewidth=5, label='Actual', alpha=0.7)
        plt.title( session_name + ' y-position prediction' , fontsize=30, color="black")

        df = pd.DataFrame( Ground_Truth )
        df.to_csv(os.path.join(csv_path, 'Ground_Truth_y_pos.csv'), index=False, header=False)

    
    plt.legend(loc='upper right', fontsize=30)
    plt.xlabel('Time (second)', fontsize=25)
    plt.ylabel('position (mm)', fontsize=25)
    plt.xticks(fontsize=25, color="black")
    plt.yticks(fontsize=25, color="black")
    axes = plt.gca()
    axes.set_xlim([time_stamp_64ms[testing_data_index]+5, time_stamp_64ms[testing_data_index]+20])
    plt.tight_layout()
    if kinematic_variable_type=='x_pos':
        plt.savefig( plot_path + '/' +model_name+'_x-position_predict.png' )
    if kinematic_variable_type=='y_pos':
        plt.savefig( plot_path + '/' +model_name+'_y-position_predict.png' )

    plt.cla()
    plt.clf()
    plt.close()

    # Save the result of kinematic variable reconstruction of each session
    df = pd.DataFrame( ((session_name, testing_data_r_square )) )
    df.to_csv(os.path.join(csv_path, 'R_square_this_session.csv'), index=False, header=False)

    df = pd.DataFrame( ((session_name, testing_data_RMSE )) )
    df.to_csv(os.path.join(csv_path, 'RMSE_this_session.csv'), index=False, header=False)

    df = pd.DataFrame( ((session_name, PCC[0] )) )
    df.to_csv(os.path.join(csv_path, 'person_correlation_coefficient__this_session.csv'), index=False, header=False)

    df = pd.DataFrame( ((session_name, testing_data_SNR )) )
    df.to_csv(os.path.join(csv_path, 'SNR_this_session.csv'), index=False, header=False)

# session control end

# Write all performances to csv files
# https://www.geeksforgeeks.org/create-a-pandas-dataframe-from-lists/
df = pd.DataFrame( list(zip( session_file_list, R_square_across_all_sessions)))
df.to_csv(os.path.join(bar_plot_path, 'R_square_across_all_sessions.csv'), index=False, header=False)

df = pd.DataFrame( list(zip( session_file_list, RMSE_across_all_sessions)))
df.to_csv(os.path.join(bar_plot_path, 'RMSE_across_all_sessions.csv'), index=False, header=False)

df = pd.DataFrame( list(zip( session_file_list, person_correlation_coefficient_across_all_sessions)))
df.to_csv(os.path.join(bar_plot_path, 'person_correlation_coefficient_across_all_sessions.csv'), index=False, header=False)

df = pd.DataFrame( list(zip( session_file_list, SNR_across_all_sessions)))
df.to_csv(os.path.join(bar_plot_path, 'SNR_across_all_sessions.csv'), index=False, header=False)


# Plot all performances as bar charts
plt.figure(figsize=(16, 9))
ind = np.arange(1,len(R_square_across_all_sessions)+1)
plt.bar(ind, R_square_across_all_sessions, width=width_two, color='r')
plt.ylabel('R square')
plt.xlabel('')
plt.xlim([0,len(R_square_across_all_sessions)+1+width_two])
plt.xticks(ind, session_file_list ,rotation=-90)
plt.grid(True)
if kinematic_variable_type=='x_pos':
    plt.title('R square of x-position prediction')
if kinematic_variable_type=='y_pos':
    plt.title('R square of y-position prediction')
plt.tight_layout()
plt.savefig(bar_plot_path+'/'+'R_square_across_sessions.png')

plt.cla()
plt.clf()
plt.close()

plt.figure(figsize=(16, 9))
ind = np.arange(1,len(RMSE_across_all_sessions)+1)
plt.bar(ind, RMSE_across_all_sessions, width=width_two, color='r')
plt.ylabel('RMSE (mm)')
plt.xlabel('')
plt.xlim([0,len(RMSE_across_all_sessions)+1+width_two])
plt.xticks(ind, session_file_list ,rotation=-90)
plt.grid(True)
if kinematic_variable_type=='x_pos':
    plt.title('RMSE of x-position prediction')
if kinematic_variable_type=='y_pos':
    plt.title('RMSE of y-position prediction')
plt.tight_layout()
plt.savefig(bar_plot_path+'/'+'RMSE_across_sessions.png')

plt.cla()
plt.clf()
plt.close()

plt.figure(figsize=(16, 9))
ind = np.arange(1,len(person_correlation_coefficient_across_all_sessions)+1)
plt.bar(ind, person_correlation_coefficient_across_all_sessions, width=width_two, color='r')
plt.ylabel('')
plt.xlabel('')
plt.xlim([0,len(person_correlation_coefficient_across_all_sessions)+1+width_two])
plt.xticks(ind, session_file_list ,rotation=-90)
plt.grid(True)
if kinematic_variable_type=='x_pos':
    plt.title('Pearson\'s correlation coefficient of x-position prediction')
if kinematic_variable_type=='y_pos':
    plt.title('Pearson\'s correlation coefficient of y-position prediction')
plt.tight_layout()
plt.savefig(bar_plot_path+'/'+'PCC_across_sessions.png')

plt.cla()
plt.clf()
plt.close()


plt.figure(figsize=(16, 9))
ind = np.arange(1,len(best_epoch_arcoss_all_sessions)+1)
plt.bar(ind, best_epoch_arcoss_all_sessions, width=width_two, color='r')
plt.ylabel('Best Epoch out of '+str(MAX_EPOCH))
plt.xlabel('')
plt.xlim([0,len(best_epoch_arcoss_all_sessions)+1+width_two])
plt.xticks(ind, session_file_list ,rotation=-90)
plt.grid(True)
if kinematic_variable_type=='x_pos':
    plt.title('Best Epoch of x-position prediction')
if kinematic_variable_type=='y_pos':
    plt.title('Best Epoch of y-position prediction')
plt.tight_layout()
plt.savefig(bar_plot_path+'/'+'best_epoch_across_sessions.png')

plt.cla()
plt.clf()
plt.close()

tEnd=time.time()
print('Overall processing time: '+ str ( round( (tEnd-tStart)/60 , 3) )+' minutes' )