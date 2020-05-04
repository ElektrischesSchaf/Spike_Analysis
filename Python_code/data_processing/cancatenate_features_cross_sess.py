import os
import numpy as np
import pandas as pd
import time
import h5py
from sklearn import datasets, svm, metrics
from sklearn.metrics import mean_squared_error, r2_score

class cross_sess_mat_file_processing():

    def cross_session_training_data_concatenation(self, feature_numbers_of_firing_rate, X, testing_data_index, X_for_training,
    x_position_label, y_position_label, z_position_label, x_velocity_label, y_velocity_label, z_velocity_label, x_acceleration_label, y_acceleration_label, z_acceleration_label,
    x_position_label_training,  y_position_label_training,  z_position_label_training, 
    x_velocity_label_training,  y_velocity_label_training,  z_velocity_label_training, 
    x_acceleration_label_training,  y_acceleration_label_training,  z_acceleration_label_training):       
      
        print('feature_numbers_of_firing_rate= ', feature_numbers_of_firing_rate, '\n')

        X_for_training = np.concatenate(( X_for_training, X[:testing_data_index, :] ), axis=0 )
        # X_for_prediction = np.concatenate(( X_for_prediction , X[testing_data_index:] ), axis=0)

        x_position_label_training = np.concatenate((x_position_label_training, x_position_label[:testing_data_index]), axis=0)
        # x_position_label_testing = np.concatenate((x_position_label_testing, x_position_label[testing_data_index:]), axis=0)

        y_position_label_training =  np.concatenate((y_position_label_training, y_position_label[:testing_data_index ]), axis=0)
        # y_position_label_testing = np.concatenate((y_position_label_testing, y_position_label[testing_data_index:]), axis=0)

        z_position_label_training = np.concatenate((z_position_label_training, z_position_label[:testing_data_index ]), axis=0)
        # z_position_label_testing = np.concatenate((z_position_label_testing, z_position_label[testing_data_index:]), axis=0)
    
        x_velocity_label_training = np.concatenate((x_velocity_label_training, x_velocity_label[:testing_data_index ]), axis=0)
        # x_velocity_label_testing = np.concatenate((x_velocity_label_testing, x_velocity_label[testing_data_index:]), axis=0)
        
        y_velocity_label_training = np.concatenate((y_velocity_label_training, y_velocity_label[:testing_data_index ] ), axis=0)
        # y_velocity_label_testing = np.concatenate((y_velocity_label_testing, y_velocity_label[testing_data_index:]), axis=0)

        z_velocity_label_training = np.concatenate((z_velocity_label_training, z_velocity_label[:testing_data_index ]), axis=0)
        # z_velocity_label_testing = np.concatenate((z_velocity_label_testing, z_velocity_label[testing_data_index:]), axis=0)

        x_acceleration_label_training = np.concatenate((x_acceleration_label_training, x_acceleration_label[:testing_data_index ]), axis=0)
        # x_acceleration_label_testing = np.concatenate((x_acceleration_label_testing, x_acceleration_label[testing_data_index:]), axis=0)

        y_acceleration_label_training = np.concatenate((y_acceleration_label_training, y_acceleration_label[:testing_data_index ]), axis=0)
        # y_acceleration_label_testing = np.concatenate((y_acceleration_label_testing, y_acceleration_label[testing_data_index:]), axis=0)

        z_acceleration_label_training = np.concatenate((z_acceleration_label_training, z_acceleration_label[:testing_data_index ]), axis=0)
        # z_acceleration_label_testing = np.concatenate((z_acceleration_label_testing, z_acceleration_label[testing_data_index:]), axis=0)

        return [X_for_training, \
        x_position_label_training, y_position_label_training, z_position_label_training, \
        x_velocity_label_training, y_velocity_label_training, z_velocity_label_training, 
        x_acceleration_label_training, y_acceleration_label_training, z_acceleration_label_training]

    def cross_session_testing_data_seperation(self, session_name, kinematic_variable_type, save_testing_data_path, X, testing_data_index, time_stamp_64ms,
    x_position_label, y_position_label, z_position_label, x_velocity_label, y_velocity_label, z_velocity_label, x_acceleration_label, y_acceleration_label, z_acceleration_label):

        save_testing_data_path=os.path.join(save_testing_data_path, 'Testing_Datasets_Folder')
        if not os.path.exists(save_testing_data_path):
            os.mkdir(save_testing_data_path)

        Testing_Datasets_Folder_path=save_testing_data_path

        save_testing_data_path=os.path.join(save_testing_data_path, session_name)
        if not os.path.exists(save_testing_data_path):
            os.mkdir(save_testing_data_path)

        # Save testing timestamp matrix
        time_stamp_64ms=time_stamp_64ms[testing_data_index:]
        df = pd.DataFrame( time_stamp_64ms )
        df.to_csv(os.path.join(save_testing_data_path, 'time_stamp_64ms_testing.csv'), index=False)   

        # Save testing firing rate matrix
        X_for_prediction =X[testing_data_index:]
        df = pd.DataFrame( X_for_prediction )
        df.to_csv(os.path.join(save_testing_data_path, 'X_for_prediction.csv'), index=False)    

        # Save testing kinematic variable matrix
        if kinematic_variable_type=='x_vel':
            # No need
            # x_velocity_label_training = x_velocity_label[:testing_data_index]
            # df = pd.DataFrame( x_velocity_label_training )
            # df.to_csv(os.path.join(save_testing_data_path, 'x_velocity_label_training.csv'), index=False)


            x_velocity_label_testing = x_velocity_label[testing_data_index:]
            df = pd.DataFrame( x_velocity_label_testing )
            df.to_csv(os.path.join(save_testing_data_path, 'x_velocity_label_testing.csv'), index=False)

        if kinematic_variable_type=='y_vel':
            # No need
            # y_velocity_label_training = y_velocity_label[:testing_data_index]
            # df = pd.DataFrame( y_velocity_label_training )
            # df.to_csv(os.path.join(save_testing_data_path, 'y_velocity_label_training.csv'), index=False)

            y_velocity_label_testing = y_velocity_label[testing_data_index:]
            df = pd.Dataframe( y_velocity_label_testing )
            df.to_csv(os.path.join(save_testing_data_path, 'y_velocity_label_testing.csv'), index=False)

        return Testing_Datasets_Folder_path