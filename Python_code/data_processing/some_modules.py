import numpy as np
import pandas as pd
import time
import h5py
import json
import os
from sklearn import datasets, svm, metrics
from sklearn.metrics import mean_squared_error, r2_score

class regular_modules():
    
    def with_or_without_sorting(self, with_sorted_spikes, firing_rate_matrix, channel_number, unit_number):
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
        return firing_rate_matrix

    def create_pathes(self, CWD,session_name, model_name, kinematic_variable_type):

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

        attention_plot_path = os.path.join(CWD, 'attention_map')
        if not os.path.exists(attention_plot_path):
            os.mkdir(attention_plot_path)


        return bar_plot_path, save_epoch_path, csv_path, plot_path, attention_plot_path

    def write_data_to_path(self, csv_path, 
    X_for_training, X_for_prediction, 
    x_position_label_training, x_position_label_testing, 
    y_position_label_training, y_position_label_testing, 
    x_velocity_label_training, x_velocity_label_testing,
    y_velocity_label_training, y_velocity_label_testing,
    x_acceleration_label_training, x_acceleration_label_testing,
    y_acceleration_label_training, y_acceleration_label_testing,
    x_position_target_training, x_position_target_testing, 
    y_position_target_training, y_position_target_testing):

        df = pd.DataFrame(X_for_training)
        df.to_csv(os.path.join(csv_path, 'trainset_feature_matrix.csv'), index=False)

        df = pd.DataFrame(X_for_prediction)
        df.to_csv(os.path.join(csv_path, 'testset_feature_matrix.csv'), index=False)
        
        # position label
        df=pd.DataFrame(x_position_label_training)
        df.to_csv(os.path.join(csv_path,'x_position_label_training.csv'), index=False)

        df=pd.DataFrame(x_position_label_testing)
        df.to_csv(os.path.join(csv_path,'x_position_label_testing.csv'), index=False)


        df=pd.DataFrame(y_position_label_training)
        df.to_csv(os.path.join(csv_path,'y_position_label_training.csv'), index=False)

        df=pd.DataFrame(y_position_label_testing)
        df.to_csv(os.path.join(csv_path,'y_position_label_testing.csv'), index=False)

        # velocity label
        df=pd.DataFrame(x_velocity_label_training)
        df.to_csv(os.path.join(csv_path,'x_velocity_label_training.csv'), index=False)

        df=pd.DataFrame(x_velocity_label_testing)
        df.to_csv(os.path.join(csv_path,'x_velocity_label_testing.csv'), index=False)

        df=pd.DataFrame(y_velocity_label_training)
        df.to_csv(os.path.join(csv_path,'y_velocity_label_training.csv'), index=False)

        df=pd.DataFrame(y_velocity_label_testing)
        df.to_csv(os.path.join(csv_path,'y_velocity_label_testing.csv'), index=False)

        # acceleration label
        df=pd.DataFrame(x_acceleration_label_training)
        df.to_csv(os.path.join(csv_path,'x_acceleration_label_training.csv'), index=False)

        df=pd.DataFrame(x_acceleration_label_testing)
        df.to_csv(os.path.join(csv_path,'x_acceleration_label_testing.csv'), index=False)

        df=pd.DataFrame(y_acceleration_label_training)
        df.to_csv(os.path.join(csv_path,'y_acceleration_label_training.csv'), index=False)

        df=pd.DataFrame(y_acceleration_label_testing)
        df.to_csv(os.path.join(csv_path,'y_acceleration_label_testing.csv'), index=False)

        # Target cue
        # df=pd.DataFrame(x_position_target_training)
        # df.to_csv(os.path.join(csv_path,'x_position_target_training.csv'), index=False)

        df=pd.DataFrame(x_position_target_testing)
        df.to_csv(os.path.join(csv_path,'x_position_target_testing.csv'), index=False)

        # df=pd.DataFrame(y_position_target_training)
        # df.to_csv(os.path.join(csv_path,'y_position_target_training.csv'), index=False)

        df=pd.DataFrame(y_position_target_testing)
        df.to_csv(os.path.join(csv_path,'y_position_target_testing.csv'), index=False)

        return