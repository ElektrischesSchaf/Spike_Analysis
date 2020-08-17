import numpy as np
import time
import h5py
from scipy.io import loadmat
import json
import os
from sklearn import datasets, svm, metrics
from sklearn.metrics import mean_squared_error, r2_score

class mat_file_processing():

    def create_empty_traing_and_testing_label(self, feature_numbers):

        X_for_training = np.empty([0, feature_numbers])
        X_for_prediction = np.empty([0, feature_numbers])
        
        x_position_label_training= np.empty([0])
        x_position_label_testing= np.empty([0])

        y_position_label_training= np.empty([0])
        y_position_label_testing= np.empty([0])

        z_position_label_training= np.empty([0])
        z_position_label_testing= np.empty([0])

        x_velocity_label_training= np.empty([0])
        x_velocity_label_testing= np.empty([0])

        y_velocity_label_training= np.empty([0])
        y_velocity_label_testing= np.empty([0])

        z_velocity_label_training= np.empty([0])
        z_velocity_label_testing= np.empty([0])

        x_acceleration_label_training= np.empty([0])
        x_acceleration_label_testing= np.empty([0])

        y_acceleration_label_training= np.empty([0])
        y_acceleration_label_testing= np.empty([0])

        z_acceleration_label_training= np.empty([0])
        z_acceleration_label_testing= np.empty([0])

        x_position_target_training= np.empty([0])
        x_position_target_testing= np.empty([0])

        y_position_target_training= np.empty([0])
        y_position_target_testing= np.empty([0])

        z_position_target_training= np.empty([0])
        z_position_target_testing= np.empty([0])

        x_velocity_target_training= np.empty([0])
        x_velocity_target_testing= np.empty([0])

        y_velocity_target_training= np.empty([0])
        y_velocity_target_testing= np.empty([0])

        z_velocity_target_training= np.empty([0])
        z_velocity_target_testing= np.empty([0])

        x_acceleration_target_training= np.empty([0])
        x_acceleration_target_testing= np.empty([0])

        y_acceleration_target_training= np.empty([0])
        y_acceleration_target_testing= np.empty([0])

        z_acceleration_target_training= np.empty([0])
        z_acceleration_target_testing= np.empty([0])

        return [X_for_training, X_for_prediction, 
        x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
        x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
        x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing,
        x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing]

    def histc(self, X, bins):
        map_to_bins = np.digitize(X,bins)
        r = np.zeros(bins.shape)
        for i in map_to_bins:
            r[i-1] += 1
        return r

    def get_spike_bins_matrix(self, the_file_name, the_sampling_rate):
        annots = loadmat(the_file_name)
        testing_data_index = 5000
        targets_corner = annots['out_struct']['targets'][0][0][0][0][0]
        targets_rotation = annots['out_struct']['targets'][0][0][0][0][1]
        pos = annots['out_struct']['pos'][0][0]
        vel = annots['out_struct']['vel'][0][0]
        acc = annots['out_struct']['acc'][0][0]

        time_stamp = pos[:,0]
        down_sampling_index = the_sampling_rate
        down_sampling_pos = pos[::down_sampling_index][:,1:]
        down_sampling_vel = vel[::down_sampling_index][:,1:]
        down_sampling_acc = acc[::down_sampling_index][:,1:]
        time_stamp_64ms = time_stamp[::down_sampling_index]

        units = annots['out_struct']['units'][0][0] # 1x174
        total_unit_numbers = units.shape[1]

        firing_rate_cell=[[]]   
        for i in range(total_unit_numbers):
            temp = units[0][i][1]
            yee = self.histc(temp, time_stamp_64ms)
            firing_rate_cell.append(yee[:-1])
            firing_rate_cell.append([])


        return [firing_rate_cell,  testing_data_index, time_stamp_64ms, total_unit_numbers]


    def get_labels(self, the_file_name, the_sampling_rate):


        annots = loadmat(the_file_name)

        targets_corner = annots['out_struct']['targets'][0][0][0][0][0]
        targets_rotation = annots['out_struct']['targets'][0][0][0][0][1]
        pos = annots['out_struct']['pos'][0][0]
        vel = annots['out_struct']['vel'][0][0]
        acc = annots['out_struct']['acc'][0][0]

        time_stamp = pos[:,0]
        down_sampling_index = the_sampling_rate
        down_sampling_pos = pos[::down_sampling_index][:,1:]
        down_sampling_vel = vel[::down_sampling_index][:,1:]
        down_sampling_acc = acc[::down_sampling_index][:,1:]
        time_stamp_64ms = time_stamp[::down_sampling_index]

        x_position_label = down_sampling_pos[1:,0]
        y_position_label = down_sampling_pos[1:,1]
        x_velocity_label = down_sampling_vel[1:,0]
        y_velocity_label = down_sampling_vel[1:,1]
        x_acceleration_label = down_sampling_acc[1:,0]
        y_acceleration_label = down_sampling_acc[1:,1]


        x_position_target = np.zeros(down_sampling_pos[1:,0].shape)

        y_position_target =  np.zeros(down_sampling_pos[1:,0].shape)

        return [time_stamp_64ms, x_position_label, y_position_label, x_velocity_label, y_velocity_label, x_acceleration_label, y_acceleration_label, x_position_target, y_position_target]

    def cross_session_data_concatenation(self, session_name, feature_numbers_of_firing_rate, X, testing_data_index, X_for_training, X_for_prediction,
    x_position_label, y_position_label, z_position_label, x_velocity_label, y_velocity_label, z_velocity_label, x_acceleration_label, y_acceleration_label, z_acceleration_label, x_position_target, y_position_target,
    x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
    x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
    x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing,
    x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing):
      
        print('feature_numbers_of_firing_rate= ', feature_numbers_of_firing_rate, '\n')

        X_for_training = np.concatenate(( X_for_training, X[:testing_data_index, :] ), axis=0 )
        X_for_prediction = np.concatenate(( X_for_prediction , X[testing_data_index:] ), axis=0)

        x_position_label_training = np.concatenate((x_position_label_training, x_position_label[:testing_data_index]), axis=0)
        x_position_label_testing = np.concatenate((x_position_label_testing, x_position_label[testing_data_index:]), axis=0)

        y_position_label_training =  np.concatenate((y_position_label_training, y_position_label[:testing_data_index ]), axis=0)
        y_position_label_testing = np.concatenate((y_position_label_testing, y_position_label[testing_data_index:]), axis=0)

        z_position_label_training = np.concatenate((z_position_label_training, z_position_label[:testing_data_index ]), axis=0)
        z_position_label_testing = np.concatenate((z_position_label_testing, z_position_label[testing_data_index:]), axis=0)
    
        x_velocity_label_training = np.concatenate((x_velocity_label_training, x_velocity_label[:testing_data_index ]), axis=0)
        x_velocity_label_testing = np.concatenate((x_velocity_label_testing, x_velocity_label[testing_data_index:]), axis=0)
        
        y_velocity_label_training = np.concatenate((y_velocity_label_training, y_velocity_label[:testing_data_index ] ), axis=0)
        y_velocity_label_testing = np.concatenate((y_velocity_label_testing, y_velocity_label[testing_data_index:]), axis=0)

        z_velocity_label_training = np.concatenate((z_velocity_label_training, z_velocity_label[:testing_data_index ]), axis=0)
        z_velocity_label_testing = np.concatenate((z_velocity_label_testing, z_velocity_label[testing_data_index:]), axis=0)

        x_acceleration_label_training = np.concatenate((x_acceleration_label_training, x_acceleration_label[:testing_data_index ]), axis=0)
        x_acceleration_label_testing = np.concatenate((x_acceleration_label_testing, x_acceleration_label[testing_data_index:]), axis=0)

        y_acceleration_label_training = np.concatenate((y_acceleration_label_training, y_acceleration_label[:testing_data_index ]), axis=0)
        y_acceleration_label_testing = np.concatenate((y_acceleration_label_testing, y_acceleration_label[testing_data_index:]), axis=0)

        z_acceleration_label_training = np.concatenate((z_acceleration_label_training, z_acceleration_label[:testing_data_index ]), axis=0)
        z_acceleration_label_testing = np.concatenate((z_acceleration_label_testing, z_acceleration_label[testing_data_index:]), axis=0)

        x_position_target_training = np.concatenate((x_position_target_training, x_position_target[:testing_data_index]), axis=0)
        x_position_target_testing = np.concatenate((x_position_target_testing, x_position_target[testing_data_index:]), axis=0)

        y_position_target_training =  np.concatenate((y_position_target_training, y_position_target[:testing_data_index ]), axis=0)
        y_position_target_testing = np.concatenate((y_position_target_testing, y_position_target[testing_data_index:]), axis=0)

        return [X_for_training, X_for_prediction, \
        x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,\
        x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,\
        x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing,\
        x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing]

    def max_order_preparation(self, session_name, order_num, feature_numbers_per_sample, 
    X_for_training, X_for_prediction,
    x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,
    x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,
    x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing,
    x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing):

        if order_num >= 2:

            order_original_matrix = X_for_training[:-order_num,:]
            for order_loop_index in range(1, order_num):
                temp_order_matrix = X_for_training[order_loop_index:-(order_num-order_loop_index),:]
                order_original_matrix = np.concatenate(( order_original_matrix, temp_order_matrix ), axis = 1)
            final_order_matrix = X_for_training[order_num:,:]
            order_original_matrix = np.concatenate(( order_original_matrix, final_order_matrix ), axis = 1)
            X_for_training = order_original_matrix.copy()

            order_original_matrix=X_for_prediction[:-order_num,:]
            for order_loop_index in range(1, order_num):
                temp_order_matrix = X_for_prediction[order_loop_index:-(order_num-order_loop_index),:]
                order_original_matrix = np.concatenate(( order_original_matrix, temp_order_matrix ), axis = 1)
            final_order_matrix = X_for_prediction[order_num:,:]
            order_original_matrix = np.concatenate(( order_original_matrix, final_order_matrix ), axis = 1)
            X_for_prediction = order_original_matrix.copy()

            x_position_label_training = x_position_label_training[order_num:]
            x_position_label_testing = x_position_label_testing[order_num:]
            y_position_label_training = y_position_label_training[order_num:]
            y_position_label_testing = y_position_label_testing[order_num:]
            z_position_label_training = z_position_label_training[order_num:]
            z_position_label_testing = z_position_label_testing[order_num:]

            x_velocity_label_training = x_velocity_label_training[order_num:]
            x_velocity_label_testing = x_velocity_label_testing[order_num:]
            y_velocity_label_training = y_velocity_label_training[order_num:]
            y_velocity_label_testing = y_velocity_label_testing[order_num:]
            z_velocity_label_training = z_velocity_label_training[order_num:]
            z_velocity_label_testing = z_velocity_label_testing[order_num:]

            x_acceleration_label_training = x_acceleration_label_training[order_num:]
            x_acceleration_label_testing = x_acceleration_label_testing[order_num:]
            y_acceleration_label_training = y_acceleration_label_training[order_num:]
            y_acceleration_label_testing = y_acceleration_label_testing[order_num:]
            z_acceleration_label_training = z_acceleration_label_training[order_num:]
            z_acceleration_label_testing = z_acceleration_label_testing[order_num:]

            x_position_target_training = x_position_target_training[order_num:]
            x_position_target_testing = x_position_target_testing[order_num:]
            y_position_target_training = y_position_target_training[order_num:]
            y_position_target_testing = y_position_target_testing[order_num:]

        aa_dict={
            "indy_20160407_02":	7777,
            "indy_20160411_01":	9894,
            "indy_20160411_02":	8749,
            "indy_20160418_01":	16212,
            "indy_20160419_01":	3187,
            "indy_20160420_01":	19268,
            "indy_20160426_01":	22532,
            "indy_20160622_01":	33276,
            "indy_20160624_03":	2812,
            "indy_20160627_01":	47546,
            "indy_20160630_01":	17863,
            "indy_20160915_01":	953,
            "indy_20160916_01":	2059,
            "indy_20160921_01":	627,
            "indy_20160927_04":	1083,
            "indy_20160927_06":	1578,
            "indy_20160930_02":	2199,
            "indy_20160930_05":	1343,
            "indy_20161005_06":	843,
            "indy_20161006_02":	2843,
            "indy_20161007_02":	2677,
            "indy_20161011_03":	5527,
            "indy_20161013_03":	3085,
            "indy_20161014_04":	3109,
            "indy_20161017_02":	2747,
            "indy_20161024_03":	2380,
            "indy_20161025_04":	2875,
            "indy_20161026_03":	2772,
            "indy_20161027_03":	4046,
            "indy_20161206_02":	6529,
            "indy_20161207_02":	1954,
            "indy_20161212_02":	3761,
            "indy_20161220_02":	4005,
            "indy_20170123_02":	4527,
            "indy_20170124_01":	4218,
            "indy_20170127_03":	6461,
            "indy_20170131_02":	7749,
            "loco_20170210_03":	2500,
            "loco_20170213_02":	26200,
            "loco_20170214_02":	5859,
            "loco_20170215_02":	12090,
            "loco_20170216_02":	7031,
            "loco_20170217_02":	5979,
            "loco_20170227_04":	25703,
            "loco_20170228_02":	15078,
            "loco_20170301_05":	4218,
            "loco_20170302_02":	17656
        }

        for session_name_from_list in  aa_dict:
            if session_name == session_name_from_list:
                length_original_testing_data = len(x_position_label_testing) + order_num
                print('length_original_testing_data = ', length_original_testing_data, '\n')
                length_difference = abs( length_original_testing_data - aa_dict[session_name] )
                print('length_difference with Makin2018 = ', length_difference, '\n')
                # start timming
                if length_difference != 0:
                    X_for_prediction = X_for_prediction[:-length_difference,:]
                    x_position_label_testing = x_position_label_testing[:-length_difference]
                    y_position_label_testing = y_position_label_testing[:-length_difference]
                    x_velocity_label_testing = x_velocity_label_testing[:-length_difference]
                    y_velocity_label_testing = y_velocity_label_testing[:-length_difference]
                    x_acceleration_label_testing = x_acceleration_label_testing[:-length_difference]
                    y_acceleration_label_testing = y_acceleration_label_testing[:-length_difference]

                    x_position_target_testing = x_position_target_testing[:-length_difference]
                    y_position_target_testing = y_position_target_testing[:-length_difference]
                break

        return [X_for_training, X_for_prediction, \
        x_position_label_training, x_position_label_testing, y_position_label_training, y_position_label_testing, z_position_label_training, z_position_label_testing,\
        x_velocity_label_training, x_velocity_label_testing, y_velocity_label_training, y_velocity_label_testing, z_velocity_label_training, z_velocity_label_testing,\
        x_acceleration_label_training, x_acceleration_label_testing, y_acceleration_label_training, y_acceleration_label_testing, z_acceleration_label_training, z_acceleration_label_testing,\
        x_position_target_training, x_position_target_testing, y_position_target_training, y_position_target_testing]