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
width_two = 0.2
# Data Processing
import pandas as pd
import json
import math
import numpy as np
from numpy import linalg as LA
import h5py
from tqdm import tqdm_notebook as tqdm
from tqdm import trange
from sklearn import datasets, svm, metrics
# Regression Problem Evaluation Methods
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
# Read/Write file
import os
CWD_origin = os.getcwd()
import shutil

import time
tStart = time.time()

# My module
import sys
sys.path.append("../../..") # Adds higher directory to python modules path.
import data_processing.parameters as my_parameters
import data_processing.load_mat_file as load_mat_file
import data_processing.load_chewie_mat_file as load_chewie_mat_file
my_parameters = my_parameters.my_parameters()
mat_file_processing = load_mat_file.mat_file_processing()
chewie_file_processing = load_chewie_mat_file.mat_file_processing()

import data_processing.some_modules as some_modules
regular_modules = some_modules.regular_modules()

# Deep leaning module
# from  Deep_Learning_Models.GRU_one_stream import GRUModel
from  Deep_Learning_Models.GRU_real_LN_no_Atten import Real_Layer_GRU_one_way
from Deep_Learning_Models.Abstract_Dataset_Class import AbstractDataset

# attention map plotting module
import Deep_Learning_Models.Attention_Map_Plotting as Attention_Map_Plotting
Plotting = Attention_Map_Plotting.Plotting()

# Make file list
kinematic_variable_type = 'x_and_y_pos' # x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc
FILE_PATH = '../../../../Dataset/Sorted_Spike_Dataset/'
ALL_List_FILE = os.listdir(FILE_PATH)
ALL_List_FILE.sort()
List_FILE = ALL_List_FILE[:]
session_file_list = List_FILE

# Neural Network Hyperparameters
model_name = 'GRU_2_outputs_one_way_real_LN_no_atten'
MAX_EPOCH = 75
LEARNING_RATE = 1e-5
NUMBER_OF_LAYERS = 2
OUTPUT_DIM = 2
BATCH_SIZE = 16
HIDDEN_DIMENSION = 256
max_timestep = 20

# Model Performance Lists
R_square_across_all_sessions=[]
SNR_across_all_sessions=[]
RMSE_across_all_sessions=[]
best_epoch_arcoss_all_sessions=[]
person_correlation_coefficient_across_all_sessions=[]
testing_data_length_all_sessions = []
my_best_epoch_dict={}

# epoch optimizer
def epoch_handle(session_name):
    epoch_dict={
    "indy_20160407_02": 31,
    "indy_20160411_01": 20,
    "indy_20160411_02": 19,
    "indy_20160418_01": 18,
    "indy_20160419_01": 37,
    "indy_20160420_01": 21,
    "indy_20160426_01": 30,
    "indy_20160622_01": 30,
    "indy_20160624_03": 33,
    "indy_20160627_01": 17,
    "indy_20160630_01": 10,
    "indy_20160915_01": 13,
    "indy_20160916_01": 12,
    "indy_20160921_01": 24,
    "indy_20160927_04": 21,
    "indy_20160927_06": 16,
    "indy_20160930_02": 32,
    "indy_20160930_05": 19,
    "indy_20161005_06": 27,
    "indy_20161006_02": 61,
    "indy_20161007_02": 52,
    "indy_20161011_03": 71,
    "indy_20161013_03": 26,
    "indy_20161014_04": 22,
    "indy_20161017_02": 46,
    "indy_20161024_03": 21,
    "indy_20161025_04": 17,
    "indy_20161026_03": 37,
    "indy_20161027_03": 11,
    "indy_20161206_02": 26,
    "indy_20161207_02": 48,
    "indy_20161212_02": 29,
    "indy_20161220_02": 77,
    "indy_20170123_02": 64,
    "indy_20170124_01": 28,
    "indy_20170127_03": 20,
    "indy_20170131_02": 49,
    "loco_20170210_03": 29,
    "loco_20170213_02": 34,
    "loco_20170214_02": 88,
    "loco_20170215_02": 20,
    "loco_20170216_02": 27,
    "loco_20170217_02": 30,
    "loco_20170227_04": 23,
    "loco_20170228_02": 110,
    "loco_20170301_05": 86,
    "loco_20170302_02": 47
    }
    for i in epoch_dict:
        if session_name==i:
            new_epoch=epoch_dict[i]
            break
    return new_epoch

# session control start
for session_k in range(len(session_file_list)):

    session_name = str(session_file_list[session_k])[:-4]

    if session_name.startswith('indy') or session_name.startswith('loco'):
        file_name_1='../../../../Dataset/Sorted_Spike_Dataset/'+ session_name +'.mat'    
        time_stamp_64ms=[]

        # Auto-assigned parameters
        testing_data_index=0
        channel_number=0
        rows_not_empty=0

        # Parameters should be assigned
        the_sampling_rate = my_parameters.the_sampling_rate
        file_numbers = my_parameters.file_numbers
        time_lag = my_parameters.time_lag

        with_sorted_spikes = True
        include_hash_unit=my_parameters.include_hash_unit

        print('In session '+ session_name + ': ' + '\n' )

        # Load Spike Firing Rate
        [firing_rate_cell, channel_number, testing_data_index, time_stamp_64ms, unit_number] = mat_file_processing.get_spike_bins_matrix(file_name_1, the_sampling_rate, time_stamp_64ms, include_hash_unit)

        # Get channel and unit numbers
        channel_numbers_in_this_dataset = channel_number
        units_numbers_in_this_dataset = unit_number


        # Eliniate empty firing rate row
        firing_rate_final=[] # not[[]]
        for row_index in range( len( firing_rate_cell) ):   
            if len(firing_rate_cell[row_index]):
                firing_rate_final.append( firing_rate_cell[row_index] )
                rows_not_empty+=1
        print('rows_not_empty = ', rows_not_empty, '\n')

        firing_rate_matrix=np.array(firing_rate_final)

        # Eliniate empty units
        valid_rows=[]
        for row_idx in range(firing_rate_matrix.shape[0]):
            if not np.all( firing_rate_matrix[row_idx,:] ==0 ):
                valid_rows.append(row_idx)
        firing_rate_matrix = firing_rate_matrix[valid_rows,:]

        print('firing_rate_matrix shape: ', firing_rate_matrix.shape) #  in indy_20160407_02 (226, 12777) eliminated null units, (288, 12777) with all 96X3 units
        print('\n')

        # get the correct sorted units number
        if with_sorted_spikes==True:
            feature_numbers = int(firing_rate_matrix.shape[0])
        else:
            feature_numbers = channel_numbers_in_this_dataset


        # Create empty arrrays from data
        [X_for_training, X_for_prediction, 
        x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
        x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
        x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing,
        x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing ]=mat_file_processing.create_empty_traing_and_testing_label(feature_numbers)

        [time_stamp_64ms, 
        x_position_label, y_position_label, z_position_label, 
        x_velocity_label, y_velocity_label, z_velocity_label, 
        x_acceleration_label, y_acceleration_label, z_acceleration_label,
        x_position_target, y_position_target] = mat_file_processing.get_labels(file_name_1, the_sampling_rate, time_stamp_64ms)




        # New Without spike sorting:
        firing_rate_matrix = regular_modules.with_or_without_sorting(with_sorted_spikes, firing_rate_matrix, channel_number, unit_number)

        firing_rate_matrix=np.transpose(firing_rate_matrix)        
        feature_numbers_of_firing_rate = firing_rate_matrix.shape[1]

        X=firing_rate_matrix.astype(np.float32)
        print('features list shape: ',end='')
        print( X.shape ) # X is the feature matrix,  (12777, 288) in indy_20160407_02
        print('\n')

    if session_name.startswith('Chewie'):

        file_name_1='../../../../Dataset/Sorted_Spike_Dataset/'+ session_name +'.mat'    
        time_stamp_64ms=[]

        # Auto-assigned parameters
        testing_data_index=0
        channel_number=0
        units_have_value=0

        # Parameters should be assigned
        the_sampling_rate = 64
        file_numbers = my_parameters.file_numbers
        time_lag = my_parameters.time_lag

        print('In session '+ session_name + ': ' + '\n' )

        # Load Spike Firing Rate
        [firing_rate_cell, testing_data_index, time_stamp_64ms, unit_number] = chewie_file_processing.get_spike_bins_matrix(file_name_1, the_sampling_rate)

        firing_rate_final=[] # not[[]]
        for row_index in range( len( firing_rate_cell) ):   
            if len(firing_rate_cell[row_index]):
                firing_rate_final.append( firing_rate_cell[row_index] )
                units_have_value+=1

        feature_numbers = unit_number

        firing_rate_matrix=np.array(firing_rate_final)
        firing_rate_matrix=np.transpose(firing_rate_matrix)        
        feature_numbers_of_firing_rate = firing_rate_matrix.shape[1]
        X=firing_rate_matrix.astype(np.float32)

        # Create empty arrrays from data
        [X_for_training, X_for_prediction, 
        x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
        x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
        x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing,
        x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing ]=chewie_file_processing.create_empty_traing_and_testing_label(feature_numbers)

        [time_stamp_64ms, 
        x_position_label, y_position_label,  
        x_velocity_label, y_velocity_label,  
        x_acceleration_label, y_acceleration_label, 
        x_position_target, y_position_target] = chewie_file_processing.get_labels(file_name_1, the_sampling_rate)


    # Cross Session Data Concatenation
    [X_for_training, X_for_prediction,
    x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, 
    x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, 
    x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing,
    x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing] = mat_file_processing.cross_session_data_concatenation(
    session_name, feature_numbers_of_firing_rate, X, testing_data_index, X_for_training, X_for_prediction,
    x_position_label, y_position_label,  x_velocity_label, y_velocity_label,  x_acceleration_label, y_acceleration_label,  x_position_target, y_position_target,
    x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, 
    x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, 
    x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing,
    x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing)
    print('shape of X_for_training before', X_for_training.shape)

    # Normalize Firing rate
    # the_mean=np.mean(X_for_training)
    # the_std=np.std(X_for_training)
    # X_for_training = (X_for_training - the_mean )/ the_std
    # X_for_prediction = (X_for_prediction - the_mean )/ the_std

    # Processing max orders
    order_num = max_timestep-1
    [X_for_training, X_for_prediction,
    x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing,  
    x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing,  
    x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing,  
    x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing] = mat_file_processing.max_order_preparation(
    session_name, order_num, feature_numbers, X_for_training, X_for_prediction,
    x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing,  
    x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing,  
    x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing,
    x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing)

    print('shape of X_for_training after', X_for_training.shape)
    print('shape of x_velocity_label_training after', x_velocity_label_training.shape)

    print('\nshape of X_for_prediction after', X_for_prediction.shape)
    testing_data_length_all_sessions.append( X_for_prediction.shape[0] )
    print('shape of x_velocity_label_testing after', x_velocity_label_testing.shape)

    # Write features and label from each session to csv files
    CWD = CWD_origin

    bar_plot_path, save_epoch_path, csv_path, plot_path, attention_plot_path = regular_modules.create_pathes(CWD, session_name, model_name, kinematic_variable_type)

    regular_modules.write_data_to_path(csv_path,
    X_for_training, X_for_prediction, 
    x_position_label_training, x_position_label_testing, 
    y_position_label_training, y_position_label_testing, 
    x_velocity_label_training, x_velocity_label_testing,
    y_velocity_label_training, y_velocity_label_testing,
    x_acceleration_label_training, x_acceleration_label_testing,
    y_acceleration_label_training, y_acceleration_label_testing,
    x_position_target_training, x_position_target_testing, 
    y_position_target_training, y_position_target_testing)


    # read from csv file
    training_x=pd.read_csv(os.path.join(csv_path, 'trainset_feature_matrix.csv'), dtype=float)
    training_x = torch.from_numpy(training_x.values) # .values can turn pandas dataframe to numpy array
    training_x=training_x.float()

    testing_x=pd.read_csv(os.path.join(csv_path, 'testset_feature_matrix.csv'), dtype=float)
    testing_x = torch.from_numpy(testing_x.values) # .values can turn pandas dataframe to numpy array
    testing_x=testing_x.float()


    # x_pos
    training_y_1 = pd.read_csv(os.path.join(csv_path,'x_position_label_training.csv'), dtype=float)    
    training_y_1  = torch.from_numpy(training_y_1.values)    
    training_y_1  = training_y_1.float()

    # y_pos
    training_y_2 = pd.read_csv(os.path.join(csv_path,'y_position_label_training.csv'), dtype=float)    
    training_y_2 = torch.from_numpy(training_y_2.values)    
    training_y_2 = training_y_2.float()


    testing_y_1 = pd.read_csv(os.path.join(csv_path,'x_position_label_testing.csv'), dtype=float)    
    testing_y_1 = torch.from_numpy(testing_y_1.values)    
    testing_y_1 = testing_y_1.float()

    testing_y_2 = pd.read_csv(os.path.join(csv_path,'y_position_label_testing.csv'), dtype=float)
    testing_y_2 = torch.from_numpy(testing_y_2.values)    
    testing_y_2 = testing_y_2.float()

    # target cue
    testing_y_3 = pd.read_csv(os.path.join(csv_path,'x_position_target_testing.csv'), dtype=float)
    testing_y_3 = torch.from_numpy(testing_y_3.values)
    testing_y_3 = testing_y_3.float()

    testing_y_4 = pd.read_csv(os.path.join(csv_path,'y_position_target_testing.csv'), dtype=float)
    testing_y_4 = torch.from_numpy(testing_y_4.values)
    testing_y_4 = testing_y_4.float()

    training_y = torch.cat( (training_y_1, training_y_2), 1)
    testing_y = torch.cat( (testing_y_1, testing_y_2), 1)

    testing_y_with_target = torch.cat( (testing_y_1, testing_y_2, testing_y_3, testing_y_4), 1)

    testing_data_length = int(testing_y.size(0))


    # General Neural Network Hyperparameters
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    max_epoch = epoch_handle(session_name) + 10

    # GRU Hyperparameters
    hidden_dim = HIDDEN_DIMENSION
    layer_dim = NUMBER_OF_LAYERS
    output_dim = OUTPUT_DIM

    # Training / Testing AbstractDataset
    training_dataset = AbstractDataset(training_x, training_y)
    testing_dataset = AbstractDataset(testing_x, testing_y)
    testing_dataset_with_target = AbstractDataset(testing_x, testing_y_with_target)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Real_Layer_GRU_one_way(input_dim = feature_numbers, hidden_dim = hidden_dim, max_timestep = max_timestep, layer_dim = layer_dim, output_dim = output_dim)     # define the network    # print(net)  # net architecture

    for n, p in net.named_parameters():
        print(n, p.shape)

    optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    net.to(device)
    history = {'train':[],'test':[]}

    def _run_epoch(epoch, mode):
        net.train(True)
        if mode == 'train':
            descrpition = 'Train'
            dataset = training_dataset
            schuffle = False
        else:
            descrpition = 'Test'
            dataset = testing_dataset
            shuffle = False
        dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                                batch_size = batch_size,
                                                shuffle = False
                                                #collate_fn = dataset.collate_fn,
                                                )
        trange = tqdm(enumerate(dataloader), total = len(dataloader), desc = descrpition)
        loss = 0

        my_prediction = []
        real_y_all = []

        for i, (x, y) in trange:

            # GRU batch
            # if(x.size()[0] is not batch_size):
            #     continue

            o_labels, batch_loss = _run_iter(x,y)

            if mode == 'train':
                optimizer.zero_grad()   # clear gradients for next train
                batch_loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients

            loss  +=  batch_loss.item() 

            real_y = y.cpu().data.numpy()
            for ele in o_labels.cpu().data.numpy():
                my_prediction.append(ele)

            for ele in real_y:
                real_y_all.append(ele)

            R_square = r2_score( real_y_all, my_prediction)

            trange.set_postfix(loss = loss/(i+1), R_square = R_square)

        if mode == 'train':
            history['train'].append({'loss':loss/len(trange), 'R^2': R_square })
            # writer.add_scalar('Loss/train', loss/len(trange), epoch)
        else:
            history['test'].append({'loss':loss/len(trange), 'R^2': R_square })
            print('uniform_average R-square: ', R_square, '\n')
            # writer.add_scalar('Loss/test', loss/len(trange), epoch)
        trange.close()

    def _run_iter(x,y):
        feature = x.to(device)
        labels = y.to(device)

        o_labels = net(feature)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        l_loss = loss_func(o_labels, labels)

        return o_labels, l_loss

    def save(epoch):
        torch.save(net.state_dict(), os.path.join( save_epoch_path, 'model.pkl.'+str(epoch) ))

        with open( os.path.join( save_epoch_path, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)

        with open( os.path.join( csv_path, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)

    for epoch in range(max_epoch):
        print('Epoch: {}'.format(epoch))
        _run_epoch(epoch, 'train')
        _run_epoch(epoch, 'test')
        save(epoch)

    # Plot the training results 
    with open(os.path.join(save_epoch_path, 'history.json'), 'r') as f:
        history = json.loads(f.read())

    regular_modules.plot_training_results( history, plot_path , model_name)

    best_score, best_epoch=max([[l['R^2'], idx] for idx, l in enumerate(history['test'])])
    print('best_score= ', best_score, ', best_epoch= ', best_epoch, '\n')    

    best_epoch_arcoss_all_sessions.append(best_epoch)
    print('Best R-square score ', max([[l['R^2'], idx] for idx, l in enumerate(history['test'])]))

    my_best_epoch_dict[session_name] = best_epoch
    
    # Testing
    best_model = best_epoch # TODO
    net.load_state_dict(state_dict = torch.load(os.path.join(save_epoch_path, 'model.pkl.{}'.format(best_model))))
    net.train(False)
    # start testing
    dataloader = DataLoader(dataset = testing_dataset_with_target,
                                batch_size = batch_size,
                                shuffle = False
                                #collate_fn = testData.collate_fn,
                                #num_workers = 8
                                )
    trange = tqdm(enumerate(dataloader), total = len(dataloader), desc = 'Predict')
    
    my_prediction_1 = []
    real_y_all_1 = []

    my_prediction_2 = []
    real_y_all_2 = []

    firing_rate_collector = []

    x_target_all = []
    y_target_all = []
    # attention map
    attn_weight_matrix_all=[]

    for i, (x, testing_y) in trange:

        # GRU batch
        # if(x.size()[0] is not batch_size):
        #     continue

        o_labels = net(x.to(device))


        # collect firing rate
        x = x.cpu().data.numpy()
        for index_batch in  range(x.shape[0]):
            firing_rate_collector.append(  x[index_batch, -feature_numbers:].flatten() )

        # collect label and target
        o_labels = o_labels.cpu().data.numpy()
        o_labels_1 = o_labels[:,0]
        o_labels_2 = o_labels[:,1]


        real_y = testing_y.cpu().data.numpy()
        real_y_1 = real_y[:,0]
        real_y_2 = real_y[:,1]
        x_target = real_y[:,2]
        y_target = real_y[:,3]

        for ele in o_labels_1:
            my_prediction_1.append( float(ele) )
        for ele in real_y_1:
            real_y_all_1.append( float(ele) )

        for ele in o_labels_2:
            my_prediction_2.append( float(ele) )
        for ele in real_y_2:
            real_y_all_2.append( float(ele) )

        for ele in x_target:
            x_target_all.append( float(ele) )
        for ele in y_target:
            y_target_all.append( float(ele) )

    shutil.rmtree(save_epoch_path)

    # collected firing rate
    firing_rate_collector = np.asarray(firing_rate_collector)
    print('shape of firing_rate_collector= ', firing_rate_collector.shape, '\n')

    testing_data_r_square_1, testing_data_SNR_1, testing_data_RMSE_1, PCC_1 = regular_modules.evlauate_performance( real_y_all_1, my_prediction_1 )
    Ground_Truth_1 = real_y_all_1
    print('\nx-position score: ', testing_data_r_square_1, ' RMSE: ', testing_data_RMSE_1, ', pearsonr=', PCC_1[0])

    testing_data_r_square_2, testing_data_SNR_2, testing_data_RMSE_2, PCC_2 = regular_modules.evlauate_performance( real_y_all_2, my_prediction_2 )
    Ground_Truth_2 = real_y_all_2
    print('\ny-position score: ', testing_data_r_square_2, ' RMSE: ', testing_data_RMSE_2, ', pearsonr=', PCC_2[0])

    R_square_across_all_sessions.append([testing_data_r_square_1, testing_data_r_square_2])
    SNR_across_all_sessions.append([testing_data_SNR_1,testing_data_SNR_2])
    RMSE_across_all_sessions.append([testing_data_RMSE_1,testing_data_RMSE_2])
    person_correlation_coefficient_across_all_sessions.append([PCC_1[0], PCC_2[0]])

    # Plotting the kinematic variable reconstructure figure
    regular_modules.kinematic_variable_reconstruction( kinematic_variable_type, model_name, '_x_and_y-position_predict.png', session_name, csv_path, plot_path, time_stamp_64ms, testing_data_index, max_timestep, my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2)

    # Save the result of kinematic variable reconstruction of each session
    df = pd.DataFrame( [[session_name, testing_data_r_square_1,  testing_data_r_square_2]], columns=['session', 'x-axis', 'y-axis'])
    df.to_csv(os.path.join(csv_path, 'R_square_this_session.csv'), index=False, header=True)

    df = pd.DataFrame( [[session_name, testing_data_RMSE_1, testing_data_RMSE_2 ]] , columns=['session', 'x-axis', 'y-axis'] )
    df.to_csv(os.path.join(csv_path, 'RMSE_this_session.csv'), index=False, header=True)

    df = pd.DataFrame( ((session_name, PCC_1[0],  PCC_2[0])) )
    df.to_csv(os.path.join(csv_path, 'person_correlation_coefficient__this_session.csv'), index=False, header=False)

    df = pd.DataFrame( [[session_name, testing_data_SNR_1, testing_data_SNR_2 ]] , columns=['session', 'x-axis', 'y-axis'] )
    df.to_csv(os.path.join(csv_path, 'SNR_this_session.csv'), index=False, header=True)



# session control end

# Write all performances to csv files
# https://www.geeksforgeeks.org/create-a-pandas-dataframe-from-lists/

regular_modules.save_across_sessions_data(bar_plot_path, session_file_list, 
R_square_across_all_sessions, RMSE_across_all_sessions, person_correlation_coefficient_across_all_sessions, SNR_across_all_sessions,
testing_data_length_all_sessions, best_epoch_arcoss_all_sessions)

with open( os.path.join( bar_plot_path, 'my_best_epoch_dict.json'), 'w') as f:
    json.dump(my_best_epoch_dict, f, indent=4)

tEnd = time.time()
print('Overall processing time: '+ str ( round( (tEnd-tStart)/60 , 3) )+' minutes' )
