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

# Make file list
kinematic_variable_type='x_vel' # x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc
FILE_PATH = '../../../Signal_Processing/Phase_all_Channels/Tables/'
ALL_List_FILE = os.listdir(FILE_PATH)
ALL_List_FILE.sort()
List_FILE=ALL_List_FILE[:]
session_file_list=List_FILE

# Neural Network Hyperparameters
model_name='Two_Stream_LSTM_with_PoF_Single_Session'
MAX_EPOCH=200
LEARNING_RATE=1e-5
NUMBER_OF_LAYERS=2
BATCH_SIZE=64
HIDDEN_DIMENSION=100

# Model Performance Lists
R_square_across_all_sessions=[]
SNR_across_all_sessions=[]
RMSE_across_all_sessions=[]
best_epoch_arcoss_all_sessions=[]
person_correlation_coefficient_across_all_sessions=[]

# session control start
for session_k in range(len(session_file_list)):
    
    session_name=str(session_file_list[session_k])
    file_name_1='../../../../Dataset/Sorted_Spike_Dataset/'+ session_name +'.mat'

    time_stamp_64ms=[]

    # import LFP Phase Data from all 96 channels start
    band_start=0.5
    band_cutoff=4
    if band_start==0.5:
        bandwidth_token='0_5' +'-'+str(band_cutoff) +'Hz'
    else:
        bandwidth_token=str(band_start)+'-'+str(band_cutoff)+'Hz'
    phase_of_firing_all_channel=[]
    for PoF_channel_index in range(0, 96):
        # TODO use np.concatenate instead
        file_phase_of_firing='../../../Signal_Processing/Phase_all_Channels/Tables/'+session_name+'/'+bandwidth_token+'/250Hz/'+str(PoF_channel_index) +'/instance_phase_a_channel_250Hz.csv'
        PoF_one_channel=pd.read_csv(file_phase_of_firing, dtype=float)
        PoF_one_channel=np.array(PoF_one_channel)
        PoF_one_channel=np.absolute(PoF_one_channel)
        # print('PoF_channel_index ', PoF_channel_index)
        # print(PoF_one_channel)
        phase_of_firing_all_channel.append( list(PoF_one_channel) )

    phase_of_firing_all_channel=np.array(phase_of_firing_all_channel)
    print( 'important= ', phase_of_firing_all_channel.shape)
    phase_of_firing_all_channel=np.squeeze(phase_of_firing_all_channel).transpose()

    # Downsampling from 250Hz to 64ms
    phase_of_firing_all_channel=phase_of_firing_all_channel[::16,:]
    # phase_of_firing_all_channel=np.reshape(96,-1)
    print( 'important= ', phase_of_firing_all_channel.shape)
    file_phase_of_firing_time_stamp='../../../Signal_Processing/Phase_all_Channels/Tables/'+session_name+'/'+bandwidth_token+'/250Hz/nwb_timestamp_to_mat_timestamp.csv'
    # import LFP Phase Data from all 96 channels end


    # Auto-assigned parameters
    testing_data_index=0
    channel_number=0
    units_have_value=0

    # Parameters should be assigned
    the_sampling_rate=my_parameters.the_sampling_rate
    file_numbers=my_parameters.file_numbers
    time_lag=my_parameters.time_lag
    order=my_parameters.order
    with_sorted_spikes=my_parameters.with_sorted_spikes
    include_hash_unit=my_parameters.include_hash_unit

    # Must know these two numbers beforehand
    channel_numbers_in_this_dataset=96
    units_numbers_in_this_dataset=3

    if with_sorted_spikes==True:
        feature_numbers=channel_numbers_in_this_dataset*units_numbers_in_this_dataset
    else:
        feature_numbers=channel_numbers_in_this_dataset
    
    # If need to concatenate Spike Firing Rate and LFP Phase, must specify the exact feature dimension here
    feature_numbers=feature_numbers*2

    [X_for_training, X_for_prediction, 
    x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
    x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
    x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing]=mat_file_processing.create_empty_traing_and_testing_label(feature_numbers)


    # cross sessions control start
    for session_index in range(file_numbers):
        print('In session '+ session_name + ': ' + '\n' )

        [firing_rate_cell, channel_number, testing_data_index, time_stamp_64ms]=mat_file_processing.get_spike_bins_matrix(file_name_1, the_sampling_rate, time_stamp_64ms, include_hash_unit)
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

        # Without spike sorting:
        if with_sorted_spikes==False:
            no_sorting_firing_rate=firing_rate_matrix.copy()
            firing_rate_matrix=np.zeros([ channel_number, firing_rate_matrix.shape[1] ])
            print('firing_rate_matrix shape: ', firing_rate_matrix.shape)  # (96, 12777)
            print('no_sorting_firing_rate shape: ', no_sorting_firing_rate.shape) # (288, 12777)
            print('\n')

            for i in range(no_sorting_firing_rate.shape[1]):
                index=0
                k=0
                while index < channel_number:
                #for k in range(channel_number-(units_numbers_in_this_dataset-1)): # Maximum 3 units in this session, indy_20160407_02.
                    #print('index: ',index,end='')
                    firing_rate_matrix[index][i]=no_sorting_firing_rate[k][i]+no_sorting_firing_rate[k+1][i]+no_sorting_firing_rate[k+2][i]

                    # Test another way to exclude hash unit, but this only works in 96 features.
                    #firing_rate_matrix[index][i]=no_sorting_firing_rate[k][i]+no_sorting_firing_rate[k+1][i]+no_sorting_firing_rate[k+2][i]

                    #print('     firing_rate_matrix[index][i]: ',firing_rate_matrix[index][i] )
                    index = index + 1
                    k = k+ units_numbers_in_this_dataset
                    #print('index: ', index, 'k: ', k)

            print('firing_rate_matrix shape: ', firing_rate_matrix.shape)  # (96, 12777)
            print('no_sorting_firing_rate shape: ', no_sorting_firing_rate.shape) # (288, 12777)
            print('\n')
        else:
            pass

        firing_rate_matrix=np.transpose(firing_rate_matrix)        
        feature_numbers_of_firing_rate = firing_rate_matrix.shape[1]

        X=firing_rate_matrix.astype(np.float32)
        print('features list shape: ',end='')
        print( X.shape ) # X is the feature matrix,  (12777, 288) in indy_20160407_02
        print('\n')

        # Adding the LFP Phase to the feature matrix

        # PoF feature matrix train / test split
        # PoF_testing_data_index=testing_data_index
        # print('PoF_testing_data_index= ',PoF_testing_data_index)
        # phase_of_firing_all_channel_traing=phase_of_firing_all_channel[:PoF_testing_data_index,:]
        # phase_of_firing_all_channel_testing=phase_of_firing_all_channel[PoF_testing_data_index:,:]

        # LFP Phase to torch
        # training_PoF=torch.from_numpy(phase_of_firing_all_channel_traing)
        # training_PoF=training_PoF.float()


        # testing_PoF=torch.from_numpy(phase_of_firing_all_channel_testing)
        # testing_PoF=testing_PoF.float()

        # Comparing the lenght of Phase-of-Firing testing dataset and Firing Rate testing dataset
        # print('phase_of_firing_all_channel_traing.size()= ', training_PoF.size(), ' training_x.size()= ',training_x.size() , '\n')
        # print('testing_PoF.size()= ', testing_PoF.size(), ' testing_x.size()= ',testing_x.size() )

        print('phase_of_firing_all_channel length = ', phase_of_firing_all_channel.shape, ' X matrix = ', X.shape)
        length_difference=phase_of_firing_all_channel.shape[0]-X.shape[0]
        print('length_difference= ', length_difference,'\n')


        # Making the lenght between Phase-of-Firing testing dataset and Firing Rate testing dataset the same
        if length_difference > 0:
            phase_of_firing_all_channel=phase_of_firing_all_channel[:-abs(length_difference),:]
        if length_difference < 0:
            print("Error in lenght of the Phase Feature")
            breakpoint()

        print('phase_of_firing_all_channel length = ', phase_of_firing_all_channel.shape, ' X matrix = ', X.shape)

        X=np.concatenate((X, phase_of_firing_all_channel) , axis=1)
        # Cancatenating Firing Rate feature matrix and Phase-of-Firing feature matrix
        # new_training_x=torch.cat(( training_x_spike, training_PoF ) , 1)
        # print('new_training_x= ', new_training_x.size())

        # new_testing_x=torch.cat(( testing_x_spike, testing_PoF), 1)

        # Above


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

    # cross sessions control end

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
    
    if kinematic_variable_type=='x_vel':
        df=pd.DataFrame(x_velocity_label_training)
        df.to_csv(os.path.join(csv_path,'x_velocity_label_training.csv'), index=False)

        df=pd.DataFrame(x_velocity_label_testing)
        df.to_csv(os.path.join(csv_path,'x_velocity_label_testing.csv'), index=False)

    if kinematic_variable_type=='y_vel':
        df=pd.DataFrame(y_velocity_label_training)
        df.to_csv(os.path.join(csv_path,'y_velocity_label_training.csv'), index=False)

        df=pd.DataFrame(y_velocity_label_testing)
        df.to_csv(os.path.join(csv_path,'y_velocity_label_testing.csv'), index=False)

    # Define AbstractDataset for Neural Networks

    class AbstractDataset(Dataset):
        def __init__(self, feature_matrix, label_matrix):
            self.data=feature_matrix
            self.label=label_matrix

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            # print('\nYee:', self.data[index], ', ' ,self.label[index])
            return self.data[index], self.label[index]
        
        def collate_fn(self, datas):
            # datas = [ batch_size X ( data + label ) ]
            print('\ncollate_fn！')
            print('\ndatas: ', datas)
            # return self.data, self.label

    # read from csv file
    training_x = pd.read_csv(os.path.join(csv_path, 'trainset_feature_matrix.csv'), dtype=float)
    training_x = torch.from_numpy(training_x.values) # .values can turn pandas dataframe to numpy array
    training_x = training_x.float()
    # training_x_spike=training_x.clone() # No need to use clone here

    testing_x=pd.read_csv(os.path.join(csv_path, 'testset_feature_matrix.csv'), dtype=float)
    testing_x = torch.from_numpy(testing_x.values) # .values can turn pandas dataframe to numpy array
    testing_x = testing_x.float()
    # testing_x_spike=testing_x.clone() # No need to use clone here

    if kinematic_variable_type=='x_vel':
        training_y=pd.read_csv(os.path.join(csv_path,'x_velocity_label_training.csv'), dtype=float)    
        training_y = torch.from_numpy(training_y.values)    
        training_y=training_y.float()

        testing_y=pd.read_csv(os.path.join(csv_path,'x_velocity_label_testing.csv'), dtype=float)    
        testing_y = torch.from_numpy(testing_y.values)    
        testing_y=testing_y.float()

    if kinematic_variable_type=='y_vel':
        training_y=pd.read_csv(os.path.join(csv_path,'y_velocity_label_training.csv'), dtype=float)    
        training_y = torch.from_numpy(training_y.values)    
        training_y=training_y.float()

        testing_y=pd.read_csv(os.path.join(csv_path,'y_velocity_label_testing.csv'), dtype=float)    
        testing_y = torch.from_numpy(testing_y.values)    
        testing_y=testing_y.float()

    '''
    for k in range(new_training_x.size(0)):
        for i in range(96):
            pass
            # new_training_x[k,-2]=abs(new_training_x[k,-2])

            # Phase-of-Firing

            # absolute
            # new_training_x[k,i]=new_training_x[k,i]*abs(new_training_x[k,-2])*new_training_x[k,-1]
            # new_training_x[k,i]=new_training_x[k,i]*abs(new_training_x[k,-2])
            # no absolute
            # new_training_x[k,i]=new_training_x[k,i]*new_training_x[k,i+96]

            # Concatenate
            # new_training_x[k,-2]=abs(new_training_x[k,-2])
            # new_training_x[k,-2]=abs(new_training_x[k,-2])*new_training_x[k,-1]
    for k in range(new_testing_x.size(0)):
        for i in range(96):
            pass
            # new_testing_x[k,-2]=abs(new_testing_x[k,-2])

            # Phase-of-Firing

            # absolute
            # new_testing_x[k,i]=new_testing_x[k,i]*abs(new_testing_x[k,-2])*new_testing_x[k,-1]
            # new_testing_x[k,i]=new_testing_x[k,i]*abs(new_testing_x[k,-2])
            # no absolute
            # new_testing_x[k,i]=new_testing_x[k,i]*new_testing_x[k,i+96]

            # Concatenate
            # new_testing_x[k,-2]=abs(new_testing_x[k,-2])
            # new_testing_x[k,-2]=abs(new_testing_x[k,-2])*new_testing_x[k,-1]

    new_training_x=new_training_x[:,:]
    new_testing_x=new_testing_x[:,:]
    '''

    real_input_features = int(training_x.size(1))
    print('first testing data time: ', time_stamp_64ms[testing_data_index], '\n')
    print('Real input features: ', real_input_features )

    # General Neural Network Hyperparameters
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    max_epoch=MAX_EPOCH

    # LSTM Hyperparameters
    hidden_dim = HIDDEN_DIMENSION
    layer_dim = NUMBER_OF_LAYERS
    output_dim = 1

    # Training / Testing AbstractDataset
    training_dataset=AbstractDataset(training_x, training_y)
    testing_dataset=AbstractDataset(testing_x, testing_y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class LSTMModel(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
            super(LSTMModel, self).__init__()
            # Hidden dimensions
            self.hidden_dim = hidden_dim
            
            # Number of hidden layers
            self.layer_dim = layer_dim
            
            # Building your LSTM
            # batch_first=True causes input/output tensors to be of shape
            # (batch_dim, seq_dim, feature_dim)
            self.lstm_spike = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
            self.lstm_phase = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
            
            # Readout layer
            self.fc1 = torch.nn.Linear(hidden_dim, int(hidden_dim/2) ) # one-directional
            self.fc2 = torch.nn.Linear(int(hidden_dim/2), output_dim) # one-directional

            # self.fc1 = torch.nn.Linear(hidden_dim*2, hidden_dim) # bidirectional
            # self.fc2 = torch.nn.Linear(hidden_dim, output_dim) # bidirectional
        def forward(self, x):

            # x torch.Size([64, 96])

            x=x.unsqueeze(0)       

            # Initialize hidden state with zeros
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_() # one-directional
            # h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_() # bidirectional
            h0=h0.to(device)
            h1=h0.clone()

            # Initialize cell state
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_() # one-directional
            # c0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_() # bidirectional
            c0=c0.to(device)
            c1=c0.clone()
            # print('input dim= ', x.size(), '\n') # input dim=  torch.Size([1, 64, 96]) => batch_first=True, (batch_dim, seq_dim, feature_dim)

            # time steps
            out_spike, (hn, cn) = self.lstm_spike(x[:,:,:96], (h0,c0))
            out_phase, (hn, cn) = self.lstm_phase(x[:,:,96:], (h1,c1))

            '''
            Index hidden state of last time step
            out.size() --> 100, 28, 100
            out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
            out = self.fc(out[:, -1, :]) 
            out.size() --> 100, 10
            '''

            out_spike=torch.relu( self.fc1(out_spike) )
            out_phase=torch.relu( self.fc1(out_phase) )
    
            out =torch.relu( self.fc1(torch.cat((out_spike,out_phase), 2) ) )
            out = self.fc2(out)
            out=out.squeeze(0)

            return out

    real_input_features=int(real_input_features /2) # Because of Two Stream LSTM

    net = LSTMModel(input_dim=real_input_features, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=output_dim)     # define the network
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

            # LSTM batch
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
        else:
            history['test'].append({'loss':loss/len(trange), 'R^2': R_square })
        trange.close()

    def _run_iter(x,y):
        feature = x.to(device)
        labels = y.to(device)
        #print('\n\n In _run_iter, ', 'shape of x', x.shape, ' ', 'shape of y', y.shape)
        o_labels = net(feature)
        #print('The output shape: ', o_labels.shape, ' The label shape: ', labels.shape, '\n')
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
    plt.title(model_name+' Loss')
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='test')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig( plot_path +'/'+model_name+'_Loss.png')

    plt.figure(figsize=(7,5))
    plt.title( model_name+ ' performance')
    plt.plot(train_R_square, label='train')
    plt.plot(valid_R_square, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('R square')
    plt.legend()
    plt.tight_layout()
    plt.savefig( plot_path+'/'+ model_name +'_R-square.png')

    #global my_prediction
    #global real_y_all


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

        # LSTM batch
        # if(x.size()[0] is not batch_size):
        #     continue

        o_labels = net(x.to(device))
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
    print('\n* model_x_velocity score: ', testing_data_r_square, ' RMSE: ', testing_data_RMSE, ', pearsonr=', PCC[0], '\n')

    R_square_across_all_sessions.append(testing_data_r_square)
    SNR_across_all_sessions.append(testing_data_SNR)
    RMSE_across_all_sessions.append(testing_data_RMSE)
    person_correlation_coefficient_across_all_sessions.append(PCC[0])

    # Plotting the kinematic variable reconstructure figure
    plt.figure(figsize=(32, 9))
    plotting_time_elapsed=time_stamp_64ms[testing_data_index:-1]

    # Ground_Truth_x_vel and Ground_Truth_y_vel may copy form testing_y in line 256 # TODO
    Ground_Truth_x_vel=x_velocity_label[testing_data_index:]
    Ground_Truth_y_vel=y_velocity_label[testing_data_index:]

    df = pd.DataFrame( plotting_time_elapsed )
    df.to_csv(os.path.join(csv_path, 'plotting_time_elapsed.csv'), index=False, header=False)

    df = pd.DataFrame( my_prediction )
    df.to_csv(os.path.join(csv_path, 'my_prediction.csv'), index=False, header=False)

    if kinematic_variable_type=='x_vel':
        plt.plot( plotting_time_elapsed, my_prediction, 'b--', linewidth=5, label='Prediction' )
        plt.plot( plotting_time_elapsed, Ground_Truth_x_vel, 'r--', linewidth=5, label='Ground Truth', alpha=0.7)
        plt.title( model_name+' Model on session: '+ session_name + ', x-velocity prediction' , fontsize=30, color="black")

        df = pd.DataFrame( Ground_Truth_x_vel )
        df.to_csv(os.path.join(csv_path, 'Ground_Truth_x_vel.csv'), index=False, header=False) 

    if kinematic_variable_type=='y_vel':
        plt.plot( plotting_time_elapsed, my_prediction, 'b--', linewidth=5, label='Prediction' )
        plt.plot( plotting_time_elapsed, Ground_Truth_y_vel, 'r--', linewidth=5, label='Ground Truth', alpha=0.7)
        plt.title( model_name+' Model on session: ' + session_name + ', y-velocity prediction' , fontsize=30, color="black")

        df = pd.DataFrame( Ground_Truth_y_vel )
        df.to_csv(os.path.join(csv_path, 'Ground_Truth_y_vel.csv'), index=False, header=False)

    
    plt.legend(loc='upper right', fontsize=30)
    plt.xlabel('Time (second)', fontsize=25)
    plt.ylabel('velocity (mm/s)', fontsize=25)
    plt.xticks(fontsize=25, color="black")
    plt.yticks(fontsize=25, color="black")
    axes = plt.gca()
    axes.set_xlim([time_stamp_64ms[testing_data_index]+5, time_stamp_64ms[testing_data_index]+20])
    plt.tight_layout()
    if kinematic_variable_type=='x_vel':
        plt.savefig( plot_path + '/' +model_name+'_x-velocity_predict.png' )
    if kinematic_variable_type=='y_vel':
        plt.savefig( plot_path + '/' +model_name+'_y-velocity_predict.png' )

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
if kinematic_variable_type=='x_vel':
    plt.title('R square of x-velocity prediction')
if kinematic_variable_type=='y_vel':
    plt.title('R square of y-velocity prediction')
plt.tight_layout()
plt.savefig(bar_plot_path+'/'+'R_square_across_sessions.png')

plt.cla()
plt.clf()
plt.close()

plt.figure(figsize=(16, 9))
ind = np.arange(1,len(RMSE_across_all_sessions)+1)
plt.bar(ind, RMSE_across_all_sessions, width=width_two, color='r')
plt.ylabel('RMSE (mm/s)')
plt.xlabel('')
plt.xlim([0,len(RMSE_across_all_sessions)+1+width_two])
plt.xticks(ind, session_file_list ,rotation=-90)
plt.grid(True)
if kinematic_variable_type=='x_vel':
    plt.title('RMSE of x-velocity prediction')
if kinematic_variable_type=='y_vel':
    plt.title('RMSE of y-velocity prediction')
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
if kinematic_variable_type=='x_vel':
    plt.title('Pearson\'s correlation coefficient of x-velocity prediction')
if kinematic_variable_type=='y_vel':
    plt.title('Pearson\'s correlation coefficient of y-velocity prediction')
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
if kinematic_variable_type=='x_vel':
    plt.title('Best Epoch of x-velocity prediction')
if kinematic_variable_type=='y_vel':
    plt.title('Best Epoch of y-velocity prediction')
plt.tight_layout()
plt.savefig(bar_plot_path+'/'+'best_epoch_across_sessions.png')

plt.cla()
plt.clf()
plt.close()

tEnd=time.time()
print('Overall processing time: '+ str ( round( (tEnd-tStart)/60 , 3) )+' minutes' )