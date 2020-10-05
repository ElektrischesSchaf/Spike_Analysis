import numpy as np
import pandas as pd
import time
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
width_two=0.2
import math
import json
import os
from sklearn import datasets, svm, metrics
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

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

        CWD = os.path.join(CWD, kinematic_variable_type)
        if not os.path.exists(CWD):
            os.mkdir(CWD)
        
        bar_plot_path = os.path.join(CWD, 'bar_plot_across_sessions')
        if not os.path.exists(bar_plot_path):
            os.mkdir(bar_plot_path)

        if session_name not in CWD:
            CWD = os.path.join(CWD, session_name)
            if not os.path.exists(CWD):
                os.mkdir(CWD)

        save_epoch_path = os.path.join(CWD,'save')
        if not os.path.exists(save_epoch_path):
            os.makedirs(save_epoch_path)

        csv_path = os.path.join(CWD,'csv_files')
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

    def plot_training_results( self, history , plot_path , model_name):
        train_loss = [l['loss'] for l in history['train']]
        valid_loss = [l['loss'] for l in history['test']]

        train_R_square = [l['R^2'] for l in history['train']]
        valid_R_square = [l['R^2'] for l in history['test']]

        plt.cla()
        plt.clf()
        plt.close()

        plt.figure(figsize=(32,18))
        plt.title('Loss', fontsize=30)
        plt.plot(train_loss, label='train')
        plt.plot(valid_loss, label='test')
        plt.xlabel('Epoch', fontsize=30)
        plt.ylabel('Loss', fontsize=30)
        plt.legend()

        plt.xticks(fontsize=25, color="black")
        plt.yticks(fontsize=25, color="black")

        plt.tight_layout()
        plt.savefig(plot_path + '/' + model_name+"_Loss.png")

        plt.cla()
        plt.clf()
        plt.close()

        plt.figure(figsize=(32,18))
        plt.title('performance', fontsize=30)
        plt.plot(train_R_square, label='train')
        plt.plot(valid_R_square, label='test')
        plt.ylim([0,1])
        plt.xlabel('Epoch', fontsize=30)
        plt.ylabel('R square', fontsize=30)
        plt.legend()

        plt.xticks(fontsize=25, color="black")
        plt.yticks(fontsize=25, color="black")

        plt.tight_layout()
        plt.savefig(plot_path + '/' +model_name+"_R-square.png")

        plt.cla()
        plt.clf()
        plt.close()

        return

    def evlauate_performance( self, real_y, predicted_y ):
        testing_data_r_square = r2_score( real_y, predicted_y )
        testing_data_SNR = -10*math.log10(1-testing_data_r_square)
        testing_data_RMSE = np.sqrt(mean_squared_error(real_y, predicted_y))
        PCC = pearsonr(real_y, predicted_y)

        return testing_data_r_square, testing_data_SNR, testing_data_RMSE, PCC

    def kinematic_variable_reconstruction(self,kinematic_variable_type,  model_name, file_name, session_name,
    csv_path, plot_path, 
    time_stamp_64ms, testing_data_index, max_timestep, 
    my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2):

        plt.figure(figsize=(32, 9))
        plotting_time_elapsed = time_stamp_64ms[testing_data_index:]
        plotting_time_elapsed = plotting_time_elapsed[max_timestep:]

        if len(plotting_time_elapsed) != len (my_prediction_1):
            diff = abs( len(plotting_time_elapsed)-len(my_prediction_1) )
            plotting_time_elapsed = plotting_time_elapsed[:-diff]

        df = pd.DataFrame( plotting_time_elapsed )
        df.to_csv(os.path.join(csv_path, 'plotting_time_elapsed.csv'), index=False, header=False)

        if kinematic_variable_type == 'x_and_y_pos':
            df = pd.DataFrame( my_prediction_1 )
            df.to_csv(os.path.join(csv_path, 'my_prediction_x_pos.csv'), index=False, header=False)

            df = pd.DataFrame( my_prediction_2 )
            df.to_csv(os.path.join(csv_path, 'my_predictiony_y_pos.csv'), index=False, header=False)

            plt.plot( plotting_time_elapsed, my_prediction_1, 'b', linewidth=5, label='x-pos prediction', alpha=0.7 )
            plt.plot( plotting_time_elapsed, Ground_Truth_1, 'b--', linewidth=5, label='x-pos actual', alpha=0.8 )

            plt.title( session_name + ', x & y position prediction' , fontsize=30, color="black")

            df = pd.DataFrame( Ground_Truth_1 )
            df.to_csv(os.path.join(csv_path, 'Ground_Truth_x_pos.csv'), index=False, header=False) 

            df = pd.DataFrame( Ground_Truth_2 )
            df.to_csv(os.path.join(csv_path, 'Ground_Truth_y_pos.csv'), index=False, header=False) 

            plt.plot( plotting_time_elapsed, my_prediction_2, 'g', linewidth=5, label='y-pos prediction', alpha=0.7 )
            plt.plot( plotting_time_elapsed, Ground_Truth_2, 'g--', linewidth=5, label='y-pos actual', alpha=0.8 )

        if kinematic_variable_type == 'x_and_y_vel':
            df = pd.DataFrame( my_prediction_1 )
            df.to_csv(os.path.join(csv_path, 'my_prediction_x_vel.csv'), index=False, header=False)

            df = pd.DataFrame( my_prediction_2 )
            df.to_csv(os.path.join(csv_path, 'my_predictiony_y_vel.csv'), index=False, header=False)

            plt.plot( plotting_time_elapsed, my_prediction_1, 'b', linewidth=5, label='x-vel prediction', alpha=0.7 )
            plt.plot( plotting_time_elapsed, Ground_Truth_1, 'b--', linewidth=5, label='x-vel actual', alpha=0.8 )

            plt.title( session_name + ', x & y velocity prediction' , fontsize=30, color="black")

            df = pd.DataFrame( Ground_Truth_1 )
            df.to_csv(os.path.join(csv_path, 'Ground_Truth_x_vel.csv'), index=False, header=False) 

            df = pd.DataFrame( Ground_Truth_2 )
            df.to_csv(os.path.join(csv_path, 'Ground_Truth_y_vel.csv'), index=False, header=False) 

            plt.plot( plotting_time_elapsed, my_prediction_2, 'g', linewidth=5, label='y-vel prediction', alpha=0.7 )
            plt.plot( plotting_time_elapsed, Ground_Truth_2, 'g--', linewidth=5, label='y-vel actual', alpha=0.8 )
    
        if kinematic_variable_type == 'x_and_y_acc':
            df = pd.DataFrame( my_prediction_1 )
            df.to_csv(os.path.join(csv_path, 'my_prediction_x_acc.csv'), index=False, header=False)

            df = pd.DataFrame( my_prediction_2 )
            df.to_csv(os.path.join(csv_path, 'my_predictiony_y_acc.csv'), index=False, header=False)

            plt.plot( plotting_time_elapsed, my_prediction_1, 'b', linewidth=5, label='x-acc prediction', alpha=0.7 )
            plt.plot( plotting_time_elapsed, Ground_Truth_1, 'b--', linewidth=5, label='x-acc actual', alpha=0.8 )

            plt.title( session_name + ', x & y acceleration prediction' , fontsize=30, color="black")

            df = pd.DataFrame( Ground_Truth_1 )
            df.to_csv(os.path.join(csv_path, 'Ground_Truth_x_acc.csv'), index=False, header=False) 

            df = pd.DataFrame( Ground_Truth_2 )
            df.to_csv(os.path.join(csv_path, 'Ground_Truth_y_acc.csv'), index=False, header=False) 

            plt.plot( plotting_time_elapsed, my_prediction_2, 'g', linewidth=5, label='y-acc prediction', alpha=0.7 )
            plt.plot( plotting_time_elapsed, Ground_Truth_2, 'g--', linewidth=5, label='y-acc actual', alpha=0.8 )
    
        plt.legend(loc='upper right', fontsize=30)
        plt.xlabel('Time (second)', fontsize=25)

        if kinematic_variable_type == 'x_and_y_pos':
            plt.ylabel('Position ($mm$)', fontsize=25)
        if kinematic_variable_type == 'x_and_y_vel':
            plt.ylabel('Velocity ($mm/s$)', fontsize=25)
        if kinematic_variable_type == 'x_and_y_acc':
            plt.ylabel('Acceleration ($mm/s^2$)', fontsize=25)

        plt.xticks(fontsize=25, color="black")
        plt.yticks(fontsize=25, color="black")
        axes = plt.gca()

        # axes.set_xlim( [   time_stamp_64ms[testing_data_index], time_stamp_64ms[testing_data_index + 230 ]  ] )
        axes.set_xlim([ plotting_time_elapsed[0], plotting_time_elapsed[0+230] ])

        plt.tight_layout()

        plt.savefig( plot_path + '/' +model_name + file_name ) # file_name 

        plt.cla()
        plt.clf()
        plt.close()

        return

    def save_across_sessions_data(self, bar_plot_path, session_file_list,
    R_square_across_all_sessions, RMSE_across_all_sessions, person_correlation_coefficient_across_all_sessions,  SNR_across_all_sessions,
    testing_data_length_all_sessions,
    best_epoch_arcoss_all_sessions):

        df = pd.DataFrame({ 'session': session_file_list, 'x-axis':[x[0] for x in R_square_across_all_sessions], 'y-axis':[x[1] for x in R_square_across_all_sessions]  })
        df.to_csv(os.path.join(bar_plot_path, 'R_square_across_all_sessions.csv'), index=False, header=True)

        df = pd.DataFrame({ 'session': session_file_list, 'x-axis':[x[0] for x in RMSE_across_all_sessions], 'y-axis':[x[1] for x in RMSE_across_all_sessions]  })
        df.to_csv(os.path.join(bar_plot_path, 'RMSE_across_all_sessions.csv'), index=False, header=True)

        df = pd.DataFrame({ 'session': session_file_list, 'x-axis':[x[0] for x in person_correlation_coefficient_across_all_sessions], 'y-axis':[x[1] for x in person_correlation_coefficient_across_all_sessions]  })
        df.to_csv(os.path.join(bar_plot_path, 'person_correlation_coefficient_across_all_sessions.csv'), index=False, header=True)

        df = pd.DataFrame({ 'session': session_file_list, 'x-axis':[x[0] for x in SNR_across_all_sessions], 'y-axis':[x[1] for x in SNR_across_all_sessions]  })
        df.to_csv(os.path.join(bar_plot_path, 'SNR_across_all_sessions.csv'), index=False, header=True)

        df = pd.DataFrame({ 'session': session_file_list, 'testing length':[x for x in testing_data_length_all_sessions] })
        df.to_csv(os.path.join(bar_plot_path, 'testing_data_length_all_sessions.csv'), index=False, header=True)

        df = pd.DataFrame({ 'session': session_file_list, 'best epoch':[x for x in best_epoch_arcoss_all_sessions] })
        df.to_csv(os.path.join(bar_plot_path, 'best_epoch_arcoss_all_sessions.csv'), index=False, header=True)

        return
