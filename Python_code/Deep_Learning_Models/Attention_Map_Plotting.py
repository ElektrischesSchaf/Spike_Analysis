import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

my_fontsize=75

class Plotting():

    # for 1-output GRU model
    def attention_map( self, end_time_bin, plot_path, my_prediction, Ground_Truth, attn_weight_matrix_all ):
        attn_weight_matrix_all=attn_weight_matrix_all[:end_time_bin,:]

        sns.set(font_scale=1.5)
        sns.set_style("white")
        sns.color_palette(palette=None)

        plt.rcParams["figure.figsize"] = (16,9)

        # grid_kws = {"height_ratios": (.3, .02, .3, .3), "hspace": 0.01}
        time=[]
        for i in range(end_time_bin):
            time+=[i]

        f, (ax, ax2) = plt.subplots(2, constrained_layout=True)

        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_all.shape[1]):
            my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )

        ax = sns.heatmap( attn_weight_matrix_all.transpose(), ax=ax, cbar=False, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False,  norm=LogNorm())
        plt.yticks(rotation=90)
        ax.set_title('Attention Map')
        ax.set_ylabel('Past Time Bins')

        ax2 = plt.plot( time, my_prediction[:end_time_bin], 'b', linewidth=3, label='Prediction' )
        ax2 = plt.plot( time, Ground_Truth[:end_time_bin], 'r', linewidth=3, label='Actual', alpha=0.7 )
        plt.legend(loc='upper right', fontsize=10)

        plt.xlim([ time[0], time[-1] ])
        plt.xlabel('Time Bin', fontsize=25)
        plt.tight_layout()
        plt.savefig( plot_path+'/'+ 'attention_map' +'.png' )

        plt.cla()
        plt.clf()
        plt.close()

        

        return


    # for 2-output GRU model
    def attention_map_2_outputs( self, session_name, time_step, type_name, start_time_bin, end_time_bin, plot_path, my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2, attn_weight_matrix_all, firing_rate_collector):

        firing_rate_collector=firing_rate_collector[start_time_bin:end_time_bin,:]
        attn_weight_matrix_all=attn_weight_matrix_all[start_time_bin:end_time_bin,:]

        sns.set(font_scale=3)
        sns.set_style("white")
        sns.color_palette(palette=None)

        plt.rcParams["figure.figsize"] = (30,30)

        time=[]
        for i in range(start_time_bin, end_time_bin):
            time+=[i]

        f, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4,3,2],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05} , constrained_layout=True)

        firing_rate_data = firing_rate_collector.transpose()

        # Eliniate empty units
        valid_rows=[]
        for row_idx in range(firing_rate_data.shape[0]):
            if not np.all( firing_rate_data[row_idx,:] ==0 ):
                valid_rows.append(row_idx)
        firing_rate_data = firing_rate_data[valid_rows,:]

        cbar_kws_firingrate = {"orientation": "horizontal", "shrink": 0.5, "aspect":40,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(firing_rate_data),  4 ]}
        sns.heatmap( data=firing_rate_data , ax=ax[0], vmax=4, cbar_kws=cbar_kws_firingrate, cmap='YlGnBu_r', yticklabels=True, xticklabels=False ) # norm=LogNorm()
    
        ax[0].set_yticklabels(ax[0].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0) # This will get correct row numbers of data matrix

        # b, t = ax[0].get_ylim() # discover the values for bottom and top
        # b += 0.5 # Add 0.5 to the bottom
        # t -= 0.5 # Subtract 0.5 from the top
        # ax[0].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[0].yaxis.set_major_locator(ticker.MultipleLocator(50))
        ax[0].yaxis.set_major_formatter(ticker.ScalarFormatter())

        # ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0) # This will get wrong row numbers of data matrix

        ax[0].set_title('Firing Rate from Session '+session_name, fontsize=my_fontsize, color="black")
        ax[0].set_ylabel('Units', fontsize=my_fontsize, color="black")


        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_all.shape[1]):
            my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )

        attention_map_data = attn_weight_matrix_all.transpose()

        cbar_kws_attention = {"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(attention_map_data), np.max(attention_map_data) ]}
        sns.heatmap( data=attention_map_data , ax=ax[1], cbar_kws=cbar_kws_attention, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False ) # norm=LogNorm()

        b, t = ax[1].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[1].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[1].set_yticklabels(ax[1].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0)
        ax[1].set_title('Attention Map', fontsize=my_fontsize, color="black")
        ax[1].set_ylabel('Past Time Bins', fontsize=my_fontsize, color="black")

        ax[2].set_title( 'Kinematic Variable Reconstruction', fontsize=my_fontsize, color="black")
        if type_name=='pos':
            ax[2].set_ylabel( 'Position (mm)', rotation=90)
        if type_name=='vel':
            ax[2].set_ylabel( 'Velocity (mm/s)', rotation=90)
        if type_name=='acc':
            ax[2].set_ylabel( 'Acceleration (mm/$s^2$)', rotation=90)
        ax[2].plot( time, my_prediction_1[start_time_bin:end_time_bin], 'b', linewidth=3, label='x-coor. prediction', alpha=0.8 )
        ax[2].plot( time, Ground_Truth_1[start_time_bin:end_time_bin], 'b--', linewidth=3, label='x-coor. actual', alpha=0.9 )

        ax[2].plot( time, my_prediction_2[start_time_bin:end_time_bin], 'g', linewidth=3, label='y-coor. prediction', alpha=0.8 )
        ax[2].plot( time, Ground_Truth_2[start_time_bin:end_time_bin], 'g--', linewidth=3, label='y-coor. actual', alpha=0.9 )
    
        ax[2].legend(loc='upper right', fontsize=my_fontsize*0.8)

        ax[2].set_xlim([ time[0], time[-1] ])
        ax[2].set_xlabel('Time Bins', fontsize=my_fontsize, color="black")

        plt.savefig( plot_path+'/'+ 'attention_map_' +str(start_time_bin)+ '_to_'+str(end_time_bin) +'.png' )

        plt.cla()
        plt.clf()
        plt.close()

        

        return

    # for 2-output GRU model
    def attention_map_2_outputs_with_target_cue( self, session_name, time_step, type_name, start_time_bin, end_time_bin, plot_path, my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2, attn_weight_matrix_all, x_target_cue, y_target_cue, firing_rate_collector):

        firing_rate_collector = firing_rate_collector[start_time_bin:end_time_bin,:]
        attn_weight_matrix_all = attn_weight_matrix_all[start_time_bin:end_time_bin,:]

        sns.set(font_scale=3)
        sns.set_style("white")
        sns.color_palette(palette=None)

        plt.rcParams["figure.figsize"] = (30,30)

        # this is use actual second in the time axes
        time_step = time_step[start_time_bin:end_time_bin]

        # This is show time bin in the time axes, not actual seconds
        # time=[]
        # for i in range(start_time_bin, end_time_bin):
        #     time+=[i]

        f, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4,3,2],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05} , constrained_layout=True)

        firing_rate_data = firing_rate_collector.transpose()

        # Eliniate empty units
        valid_rows=[]
        for row_idx in range(firing_rate_data.shape[0]):
            if not np.all( firing_rate_data[row_idx,:] ==0 ):
                valid_rows.append(row_idx)
        firing_rate_data = firing_rate_data[valid_rows,:]

        cbar_kws_firingrate = {"orientation": "horizontal", "shrink": 0.5, "aspect":40,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(firing_rate_data),  4 ]}
        sns.heatmap( data=firing_rate_data , ax=ax[0], vmax=4, cbar_kws=cbar_kws_firingrate, cmap='YlGnBu_r', yticklabels=True, xticklabels=False ) # norm=LogNorm()
    
        ax[0].set_yticklabels(ax[0].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0) # This will get correct row numbers of data matrix

        # b, t = ax[0].get_ylim() # discover the values for bottom and top
        # b += 0.5 # Add 0.5 to the bottom
        # t -= 0.5 # Subtract 0.5 from the top
        # ax[0].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[0].yaxis.set_major_locator(ticker.MultipleLocator(50))
        ax[0].yaxis.set_major_formatter(ticker.ScalarFormatter())

        # ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0) # This will get wrong row numbers of data matrix

        ax[0].set_title('Firing Rate from Session '+session_name, fontsize=my_fontsize, color="black")
        ax[0].set_ylabel('Units', fontsize=my_fontsize, color="black")


        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_all.shape[1]):
            my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )

        attention_map_data = attn_weight_matrix_all.transpose()

        cbar_kws_attention = {"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(attention_map_data), np.max(attention_map_data) ]}
        sns.heatmap( data=attention_map_data , ax=ax[1], cbar_kws=cbar_kws_attention, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False ) # norm=LogNorm()

        b, t = ax[1].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[1].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[1].set_yticklabels(ax[1].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0)
        ax[1].set_title('Attention Map', fontsize=my_fontsize, color="black")
        ax[1].set_ylabel('Past Time Bins', fontsize=my_fontsize, color="black")

        ax[2].set_title( 'Kinematic Variable Reconstruction', fontsize=my_fontsize, color="black")
        if type_name=='pos':
            ax[2].set_ylabel( 'Position (mm)', rotation=90)
        if type_name=='vel':
            ax[2].set_ylabel( 'Velocity (mm/s)', rotation=90)
        if type_name=='acc':
            ax[2].set_ylabel( 'Acceleration (mm/$s^2$)', rotation=90)
        ax[2].plot( time_step, my_prediction_1[start_time_bin:end_time_bin], 'b', linewidth=3, label='x-coor. prediction', alpha=0.8 )
        ax[2].plot( time_step, Ground_Truth_1[start_time_bin:end_time_bin], 'b--', linewidth=3, label='x-coor. actual', alpha=0.9 )

        ax[2].plot( time_step, my_prediction_2[start_time_bin:end_time_bin], 'g', linewidth=3, label='y-coor. prediction', alpha=0.8 )
        ax[2].plot( time_step, Ground_Truth_2[start_time_bin:end_time_bin], 'g--', linewidth=3, label='y-coor. actual', alpha=0.9 )

        # plot target cue change points
        the_x = x_target_cue[start_time_bin:end_time_bin]
        the_y = y_target_cue[start_time_bin:end_time_bin]
        change_points_x = np.where(  np.roll( the_x,1)!= the_x )[0]
        change_points_y = np.where( np.roll( the_y,1)!= the_y )[0]

        change_points_x_set = set(change_points_x)
        change_points_y_set = set(change_points_y)
        
        # seconds version
        for ele in list(change_points_x_set.union( change_points_y_set )):
            ax[2].axvline( time_step[ele] , color='black' , linewidth=5, alpha=0.3 )
        
        # time bin version
        # for ele in list(change_points_x_set.union( change_points_y_set )):
        #     ax[2].axvline( time[0]+ele , color='black' , linewidth=5, alpha=0.3 )

        # this is for target cue checking, time bin version
        # ax[2].plot(time, x_target_cue[start_time_bin:end_time_bin] , 'ob', linewidth=3, label='x-coor. cue', alpha=0.8)
        # ax[2].plot(time, y_target_cue[start_time_bin:end_time_bin], 'og',linewidth=3, label='y-coor. cue', alpha=0.8)


        ax[2].legend(loc='upper right', fontsize=my_fontsize*0.8)


        # ax[2].set_xlim([ time[0], time[-1] ])
        # ax[2].set_xlabel('Time Bins', fontsize=my_fontsize, color="black")

        ax[2].set_xlim([ time_step[0], time_step[-1] ])
        ax[2].set_xlabel('Time (seconds)', fontsize=my_fontsize, color="black")

        plt.savefig( plot_path+'/'+ 'attention_map_' +str(start_time_bin)+ '_to_'+str(end_time_bin) +'.png' )

        plt.cla()
        plt.clf()
        plt.close()

        return

    # for new try
    def new_atten_map_try( self, session_name, time_step, type_name, start_time_bin, end_time_bin, plot_path, my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2, attn_weight_matrix_all, x_target_cue, y_target_cue, firing_rate_collector, q_value_all, loss_plot_vector_x, loss_plot_vector_y):
        
        firing_rate_collector = firing_rate_collector[start_time_bin:end_time_bin,:]
        attn_weight_matrix_all = attn_weight_matrix_all[start_time_bin:end_time_bin,:]

        sns.set(font_scale=5)
        sns.set_style("white")
        sns.color_palette(palette=None)

        plt.rcParams["figure.figsize"] = (35,20)

        # this is use actual second in the time axes
        time_step = time_step[start_time_bin:end_time_bin]

        f, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [1,1,1,1],  "hspace":0.1 ,"left":0.1, "right":0.95, "top":0.95, "bottom":0.1} , constrained_layout=False)

        firing_rate_data = firing_rate_collector.transpose()

        # Eliniate empty units
        valid_rows=[]
        for row_idx in range(firing_rate_data.shape[0]):
            if not np.all( firing_rate_data[row_idx,:] ==0 ):
                valid_rows.append(row_idx)
        firing_rate_data = firing_rate_data[valid_rows,:]


        # Plotting the absolute velocity
        if type_name=='x_and_y_pos':
            ax[0].set_ylabel( 'mm/s', rotation=90, fontsize=my_fontsize, color="black")

            velocity_x = np.diff(Ground_Truth_1[start_time_bin:end_time_bin])
            velocity_y = np.diff(Ground_Truth_2[start_time_bin:end_time_bin])

            velocity_x = np.insert(velocity_x, 0, 0)
            velocity_y = np.insert(velocity_y, 0, 0)

            velocity_x =velocity_x/0.064
            velocity_y =velocity_y/0.064

            velocity_absolute = []

            for i in range(len(velocity_x)):
                yee = np.sqrt( velocity_x[i]*velocity_x[i] + velocity_y[i]*velocity_y[i] )
                velocity_absolute.append( yee )

        if type_name== 'x_and_y_vel':
            ax[0].set_ylabel( 'mm/s', rotation=90, fontsize=my_fontsize, color="black")

            velocity_x = Ground_Truth_1[start_time_bin:end_time_bin]
            velocity_y = Ground_Truth_2[start_time_bin:end_time_bin]

            velocity_absolute = []

            for i in range(len(velocity_x)):
                yee = np.sqrt( velocity_x[i]*velocity_x[i] + velocity_y[i]*velocity_y[i] )
                velocity_absolute.append( yee )

        ax[0].plot( time_step, velocity_absolute, 'b', linewidth=3, label='absolute velocity', alpha=1.0 )

        # plot target cue change points
        the_x = x_target_cue[start_time_bin:end_time_bin]
        the_y = y_target_cue[start_time_bin:end_time_bin]
        change_points_x = np.where(  np.roll( the_x,1)!= the_x )[0]
        change_points_y = np.where( np.roll( the_y,1)!= the_y )[0]

        change_points_x_set = set(change_points_x)
        change_points_y_set = set(change_points_y)
        
        # seconds version
        for ele in list(change_points_x_set.union( change_points_y_set )):
            ax[0].axvline( time_step[ele] , color='black' , linewidth=5, alpha=0.3 )

        ax[0].legend(loc='upper right', fontsize=my_fontsize*0.5)
        ax[0].tick_params(axis='both', which='major', labelsize=my_fontsize*0.7, color="black")
        ax[0].set_xlim([ time_step[0], time_step[-1] ])
        ax[0].set_ylim([ 0,  500 ])
        # ax[0].set_xlabel('Time (seconds)', fontsize=my_fontsize, color="black")
        ax[0].set_xticks([])


        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_all.shape[1]):
            if ticks_label == 0:
                my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )
            elif ticks_label == int(attn_weight_matrix_all.shape[1]) -1 :
                my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )
            elif ticks_label == int( int(attn_weight_matrix_all.shape[1]) / 2 ):
                my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )
            else:
                my_yticklabels.append('')


        # Plotting attention maps
        attention_map_data = attn_weight_matrix_all.transpose()

        cbar_kws_attention = {"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(attention_map_data), np.max(attention_map_data) ]}
        sns.heatmap( data=attention_map_data, ax=ax[1], cbar_kws=cbar_kws_attention, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False , cbar=False) # norm=LogNorm()
        
        b, t = ax[1].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[1].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[1].set_yticklabels(ax[1].get_ymajorticklabels(), fontsize = my_fontsize*0.7, rotation=0)

        # ax[1].set_title('Attention Map', fontsize=my_fontsize, color="black")
        ax[1].set_ylabel('taps', fontsize=my_fontsize, color="black")


        ax_new=ax[2].twinx()
        # Plotting spike counts
        firing_rate_tota_counts = firing_rate_data.sum(axis=0)
        curve1, = ax[2].plot( time_step, firing_rate_tota_counts, 'r', linewidth=3, label='spike counts', alpha=1.0 )
        ax[2].set_xlim([ time_step[0], time_step[-1] ])
        ax[2].set_ylim([ 0,  350 ])
        ax[2].set_xticks([])
        # ax[2].set_ylabel( 'counts', rotation=90, fontsize=my_fontsize, color="black")
        ax[2].tick_params(axis='y', labelcolor='tab:red')
        # ax[2].legend(loc='upper right', fontsize=my_fontsize*0.5)
        

        # Plotting attention map max index
        attention_map_max_index = np.argmax(attention_map_data, axis=0)
        curve2, = ax_new.plot( time_step, attention_map_max_index*-1 , 'b', linewidth=3, label='attention map', alpha=1.0 )
        # ax[2].set_xlim([ time_step[0], time_step[-1] ])
        # ax[2].set_ylim([ 0,  350 ])
        ax_new.set_yticks([])
        ax_new.tick_params(axis='y', labelcolor='tab:blue')

        curves = [curve1, curve2]
        ax_new.legend( curves, [curve.get_label() for curve in curves ], loc='upper right', fontsize=my_fontsize*0.5)


        num_ticks = 6
        my_second_labels = time_step
        # the index of the position of yticks
        xticks = np.linspace(0, len(my_second_labels) - 1, num_ticks, dtype=np.int)
        # the content of labels of these yticks
        # xticklabels = [ round(my_second_labels[idx], 2) for idx in xticks ] # float version
        xticklabels = [ int(my_second_labels[idx]) for idx in xticks ] # int version

        cbar_kws_firingrate = {"orientation": "horizontal", "shrink": 0.5, "aspect":40,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(firing_rate_data),  4 ]}
        sns.heatmap( data=firing_rate_data , ax=ax[3], vmax=4, cbar_kws=cbar_kws_firingrate, cmap='YlGnBu_r', yticklabels=True, xticklabels=xticklabels, cbar=False ) # norm=LogNorm()

        ax[3].set_xticklabels(ax[3].get_xmajorticklabels(), fontsize = my_fontsize*0.7 , rotation=0)
        ax[3].set_yticklabels(ax[3].get_ymajorticklabels(), fontsize = my_fontsize*0.7 , rotation=0) # This will get correct row numbers of data matrix

        ax[3].set_xticks(xticks)

        ax[3].yaxis.set_major_locator(ticker.MultipleLocator(100))
        ax[3].yaxis.set_major_formatter(ticker.ScalarFormatter())        

        
        ax[3].set_ylabel('units', fontsize=my_fontsize, color="black")
        ax[3].set_xlabel('time (s)', fontsize=my_fontsize, color="black")

        plt.savefig( plot_path+'/'+ 'attention_map_' +str(start_time_bin)+ '_to_'+str(end_time_bin) +'.png' )


        plt.cla()
        plt.clf()
        plt.close()
    
        return

    # for the report
    def attention_map_2_outputs_with_target_cue_no_colorbar( self, session_name, time_step, type_name, start_time_bin, end_time_bin, plot_path, my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2, attn_weight_matrix_all, x_target_cue, y_target_cue, firing_rate_collector, q_value_all, loss_plot_vector_x, loss_plot_vector_y):

        firing_rate_collector = firing_rate_collector[start_time_bin:end_time_bin,:]
        attn_weight_matrix_all = attn_weight_matrix_all[start_time_bin:end_time_bin,:]

        sns.set(font_scale=5)
        sns.set_style("white")
        sns.color_palette(palette=None)

        plt.rcParams["figure.figsize"] = (35,20)

        # this is use actual second in the time axes
        time_step = time_step[start_time_bin:end_time_bin]

        f, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1,1,1],  "hspace":0.1 ,"left":0.1, "right":0.95, "top":0.95, "bottom":0.1} , constrained_layout=False)

        firing_rate_data = firing_rate_collector.transpose()

        # Eliniate empty units
        valid_rows=[]
        for row_idx in range(firing_rate_data.shape[0]):
            if not np.all( firing_rate_data[row_idx,:] ==0 ):
                valid_rows.append(row_idx)
        firing_rate_data = firing_rate_data[valid_rows,:]

        # ax[0].set_title( 'Kinematic Variable Reconstruction', fontsize=my_fontsize, color="black")
        if type_name=='x_and_y_pos':
            ax[0].set_ylabel( 'pos. (mm)', rotation=90, fontsize=my_fontsize, color="black")
        if type_name=='x_and_y_vel':
            ax[0].set_ylabel( 'vel. (mm/s)', rotation=90, fontsize=my_fontsize, color="black")
        if type_name=='x_and_y_acc':
            ax[0].set_ylabel( 'acc. (mm/$s^2$)', rotation=90, fontsize=my_fontsize*0.7, color="black")

        ax[0].plot( time_step, my_prediction_1[start_time_bin:end_time_bin], 'b', linewidth=3, label='x-coor. prediction', alpha=0.8 )
        ax[0].plot( time_step, Ground_Truth_1[start_time_bin:end_time_bin], 'b--', linewidth=3, label='x-coor. actual', alpha=0.9 )

        ax[0].plot( time_step, my_prediction_2[start_time_bin:end_time_bin], 'g', linewidth=3, label='y-coor. prediction', alpha=0.8 )
        ax[0].plot( time_step, Ground_Truth_2[start_time_bin:end_time_bin], 'g--', linewidth=3, label='y-coor. actual', alpha=0.9 )

        # plot target cue change points
        the_x = x_target_cue[start_time_bin:end_time_bin]
        the_y = y_target_cue[start_time_bin:end_time_bin]
        change_points_x = np.where(  np.roll( the_x,1)!= the_x )[0]
        change_points_y = np.where( np.roll( the_y,1)!= the_y )[0]

        change_points_x_set = set(change_points_x)
        change_points_y_set = set(change_points_y)
        
        # seconds version
        for ele in list(change_points_x_set.union( change_points_y_set )):
            ax[0].axvline( time_step[ele] , color='black' , linewidth=5, alpha=0.3 )

        ax[0].legend(loc='upper right', fontsize=my_fontsize*0.5)
        ax[0].tick_params(axis='both', which='major', labelsize=my_fontsize*0.7, color="black")
        ax[0].set_xlim([ time_step[0], time_step[-1] ])
        # ax[0].set_xlabel('Time (seconds)', fontsize=my_fontsize, color="black")
        ax[0].set_xticks([])


        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_all.shape[1]):
            if ticks_label == 0:
                my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )
            elif ticks_label == int(attn_weight_matrix_all.shape[1]) -1 :
                my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )
            elif ticks_label == int( int(attn_weight_matrix_all.shape[1]) / 2 ):
                my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )
            else:
                my_yticklabels.append('')

        attention_map_data = attn_weight_matrix_all.transpose()

        cbar_kws_attention = {"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(attention_map_data), np.max(attention_map_data) ]}
        sns.heatmap( data=attention_map_data, ax=ax[1], cbar_kws=cbar_kws_attention, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False , cbar=False) # norm=LogNorm()
        
        b, t = ax[1].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[1].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[1].set_yticklabels(ax[1].get_ymajorticklabels(), fontsize = my_fontsize*0.7, rotation=0)

        # ax[1].set_title('Attention Map', fontsize=my_fontsize, color="black")
        ax[1].set_ylabel('taps', fontsize=my_fontsize, color="black")


        num_ticks = 6
        my_second_labels = time_step
        # the index of the position of yticks
        xticks = np.linspace(0, len(my_second_labels) - 1, num_ticks, dtype=np.int)
        # the content of labels of these yticks
        # xticklabels = [ round(my_second_labels[idx], 2) for idx in xticks ] # float version
        xticklabels = [ int(my_second_labels[idx]) for idx in xticks ] # int version

        cbar_kws_firingrate = {"orientation": "horizontal", "shrink": 0.5, "aspect":40,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(firing_rate_data),  4 ]}
        sns.heatmap( data=firing_rate_data , ax=ax[2], vmax=4, cbar_kws=cbar_kws_firingrate, cmap='YlGnBu_r', yticklabels=True, xticklabels=xticklabels, cbar=False ) # norm=LogNorm()

        ax[2].set_xticklabels(ax[2].get_xmajorticklabels(), fontsize = my_fontsize*0.7 , rotation=0)
        ax[2].set_yticklabels(ax[2].get_ymajorticklabels(), fontsize = my_fontsize*0.7 , rotation=0) # This will get correct row numbers of data matrix

        ax[2].set_xticks(xticks)

        ax[2].yaxis.set_major_locator(ticker.MultipleLocator(100))
        ax[2].yaxis.set_major_formatter(ticker.ScalarFormatter())        

        # ax[2].set_title('Firing Rate from Session '+session_name, fontsize=my_fontsize, color="black")
        ax[2].set_ylabel('units', fontsize=my_fontsize, color="black")
        ax[2].set_xlabel('time (s)', fontsize=my_fontsize, color="black")

        plt.savefig( plot_path+'/'+ 'attention_map_' +str(start_time_bin)+ '_to_'+str(end_time_bin) +'.png' )


        plt.cla()
        plt.clf()
        plt.close()
    
        return


    
    def quant( self, session_name, time_step, type_name, start_time_bin, end_time_bin, plot_path, my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2, attn_weight_matrix_all, x_target_cue, y_target_cue, firing_rate_collector, q_value_all, loss_plot_vector_x, loss_plot_vector_y):

        firing_rate_collector = firing_rate_collector[start_time_bin:end_time_bin,:]
        attn_weight_matrix_all = attn_weight_matrix_all[start_time_bin:end_time_bin,:]
        q_value_all = q_value_all[start_time_bin:end_time_bin,:]
        q_value_all = q_value_all.transpose()

        loss_plot_vector_x = loss_plot_vector_x[start_time_bin:end_time_bin,:]
        loss_plot_vector_x = loss_plot_vector_x.transpose()

        loss_plot_vector_y = loss_plot_vector_y[start_time_bin:end_time_bin,:]
        loss_plot_vector_y = loss_plot_vector_y.transpose()

        sns.set(font_scale=3)
        sns.set_style("white")
        sns.color_palette(palette=None)

        plt.rcParams["figure.figsize"] = (30,20)

        # this is use actual second in the time axes
        time_step = time_step[start_time_bin:end_time_bin]

        f, ax = plt.subplots(5, 1, gridspec_kw={'height_ratios': [10, 10, 2, 2, 10],  "hspace":0.1 ,"left":0.1, "right":0.95, "top":0.95, "bottom":0.1} , constrained_layout=False)

        firing_rate_data = firing_rate_collector.transpose()

        # Eliniate empty units
        valid_rows=[]
        for row_idx in range(firing_rate_data.shape[0]):
            if not np.all( firing_rate_data[row_idx,:] ==0 ):
                valid_rows.append(row_idx)
        firing_rate_data = firing_rate_data[valid_rows,:]

        # ax[0].set_title( 'Kinematic Variable Reconstruction', fontsize=my_fontsize, color="black")
        if type_name=='x_and_y_pos':
            ax[0].set_ylabel( 'pos (mm)', rotation=90, fontsize=my_fontsize, color="black")
        if type_name=='x_and_y_vel':
            ax[0].set_ylabel( 'vel (mm/s)', rotation=90, fontsize=my_fontsize, color="black")
        if type_name=='x_and_y_acc':
            ax[0].set_ylabel( 'acc (mm/$\mathrm{s}^{\mathrm{2}}$)', rotation=90, fontsize=my_fontsize, color="black")

        ax[0].plot( time_step, my_prediction_1[start_time_bin:end_time_bin], 'b', linewidth=3, label='x-coor. prediction', alpha=0.8 )
        ax[0].plot( time_step, Ground_Truth_1[start_time_bin:end_time_bin], 'b--', linewidth=3, label='x-coor. actual', alpha=0.9 )

        ax[0].plot( time_step, my_prediction_2[start_time_bin:end_time_bin], 'g', linewidth=3, label='y-coor. prediction', alpha=0.8 )
        ax[0].plot( time_step, Ground_Truth_2[start_time_bin:end_time_bin], 'g--', linewidth=3, label='y-coor. actual', alpha=0.9 )

        # plot target cue change points
        the_x = x_target_cue[start_time_bin:end_time_bin]
        the_y = y_target_cue[start_time_bin:end_time_bin]
        change_points_x = np.where(  np.roll( the_x,1)!= the_x )[0]
        change_points_y = np.where( np.roll( the_y,1)!= the_y )[0]

        change_points_x_set = set(change_points_x)
        change_points_y_set = set(change_points_y)
        
        # seconds version
        for ele in list(change_points_x_set.union( change_points_y_set )):
            ax[0].axvline( time_step[ele] , color='black' , linewidth=5, alpha=0.3 )

        ax[0].legend(loc='upper right', fontsize=my_fontsize*0.5)
        ax[0].tick_params(axis='both', which='major', labelsize=my_fontsize*0.7, color="black")
        ax[0].set_xlim([ time_step[0], time_step[-1] ])
        # ax[0].set_xlabel('Time (seconds)', fontsize=my_fontsize, color="black")
        ax[0].set_xticks([])


        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_all.shape[1]):
            if ticks_label == 0:
                my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )
            elif ticks_label == int(attn_weight_matrix_all.shape[1]) -1 :
                # my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )
                my_yticklabels.append( 't')
            elif ticks_label == int( int(attn_weight_matrix_all.shape[1]) / 2 ):
                my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )
            else:
                my_yticklabels.append('')

        attention_map_data = attn_weight_matrix_all.transpose()

        cbar_kws_attention = {"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(attention_map_data), np.max(attention_map_data) ]}
        
        sns.heatmap( data=attention_map_data, ax=ax[1], cbar_kws=cbar_kws_attention, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False , cbar=False) # norm=LogNorm()
        
        b, t = ax[1].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[1].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[1].set_yticklabels(ax[1].get_ymajorticklabels(), fontsize = my_fontsize*0.7, rotation=0)

        # ax[1].set_title('Attention Map', fontsize=my_fontsize, color="black")
        ax[1].set_ylabel('taps', fontsize=my_fontsize, color="black")



        # Q VALUE
        # cbar_kws_firingrate = {"orientation": "horizontal", "shrink": 1 , "aspect": 40 ,"use_gridspec":"True", "fraction":0.15 , "pad":0.1, 'ticks' : [ 0 , np.max(q_value_all)]  }        
        # sns.heatmap( data = q_value_all, ax=ax[2], vmin=0, cbar_kws=cbar_kws_firingrate, cmap='Blues', yticklabels=False, xticklabels=False, cbar=False ) # norm=LogNorm()

        # b, t = ax[2].get_ylim() # discover the values for bottom and top
        # b += 0.5 # Add 0.5 to the bottom
        # t -= 0.5 # Subtract 0.5 from the top
        # ax[2].set_ylim(b, t) # update the ylim(bottom, top) values

        # ax[2].set_ylabel( 'STD' , fontsize=my_fontsize*0.6, rotation=0, color="black", labelpad= 70 )


        # x-coor. error
        cbar_kws_firingrate = {"orientation": "horizontal", "shrink": 0.5, "aspect":40,"use_gridspec":"True", "fraction":0.01 , "pad":0.03 }
        if type_name == 'x_and_y_pos':
            sns.heatmap( data =  loss_plot_vector_x, vmin=0, vmax=60, ax=ax[2], cbar_kws=cbar_kws_firingrate, cmap='Reds', yticklabels=False, xticklabels=False, cbar=False ) # norm=LogNorm()
        if type_name == 'x_and_y_vel':
            sns.heatmap( data =  loss_plot_vector_x, vmin=0, vmax=200, ax=ax[2], cbar_kws=cbar_kws_firingrate, cmap='Reds', yticklabels=False, xticklabels=False, cbar=False ) # norm=LogNorm()
        if type_name == 'x_and_y_acc':
            sns.heatmap( data =  loss_plot_vector_x, vmin=0, vmax=2000, ax=ax[2], cbar_kws=cbar_kws_firingrate, cmap='Reds', yticklabels=False, xticklabels=False, cbar=False ) # norm=LogNorm()

        b, t = ax[2].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[2].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[2].set_ylabel( 'x error' , fontsize=my_fontsize*0.6, rotation=0, color="black", labelpad= 70 )

        # y-coor. error
        cbar_kws_firingrate = {"orientation": "horizontal", "shrink": 0.5, "aspect":40,"use_gridspec":"True", "fraction":0.01 , "pad":0.03 }
        if type_name == 'x_and_y_pos':
            sns.heatmap( data =  loss_plot_vector_y, vmin=0, vmax=60, ax=ax[3], cbar_kws=cbar_kws_firingrate, cmap='Reds', yticklabels=False, xticklabels=False, cbar=False ) # norm=LogNorm()
        if type_name == 'x_and_y_vel':
            sns.heatmap( data =  loss_plot_vector_y, vmin=0, vmax=200, ax=ax[3], cbar_kws=cbar_kws_firingrate, cmap='Reds', yticklabels=False, xticklabels=False, cbar=False ) # norm=LogNorm()            
        if type_name == 'x_and_y_acc':
            sns.heatmap( data =  loss_plot_vector_y, vmin=0, vmax=2000, ax=ax[3], cbar_kws=cbar_kws_firingrate, cmap='Reds', yticklabels=False, xticklabels=False, cbar=False ) # norm=LogNorm()

        b, t = ax[3].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[3].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[3].set_ylabel( 'y error' , fontsize=my_fontsize*0.6, rotation=0, color="black", labelpad= 70 )

        # FIRING RATE
        num_ticks = 6
        my_second_labels = time_step
        # the index of the position of yticks
        xticks = np.linspace(0, len(my_second_labels) - 1, num_ticks, dtype=np.int)
        # the content of labels of these yticks
        # xticklabels = [ round(my_second_labels[idx], 2) for idx in xticks ] # float version
        xticklabels = [ int(my_second_labels[idx]) for idx in xticks ] # int version

        cbar_kws_firingrate = {"orientation": "horizontal", "shrink": 0.5, "aspect":40,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(firing_rate_data),  4 ]}
        sns.heatmap( data=firing_rate_data , ax=ax[4], vmax=4, cbar_kws=cbar_kws_firingrate, cmap='YlGnBu_r', yticklabels=True, xticklabels=xticklabels, cbar=False ) # norm=LogNorm()

        ax[4].set_xticklabels(ax[4].get_xmajorticklabels(), fontsize = my_fontsize*0.7 , rotation=0)
        ax[4].set_yticklabels(ax[4].get_ymajorticklabels(), fontsize = my_fontsize*0.7 , rotation=0) # This will get correct row numbers of data matrix

        ax[4].set_xticks(xticks)

        ax[4].yaxis.set_major_locator(ticker.MultipleLocator(100))
        ax[4].yaxis.set_major_formatter(ticker.ScalarFormatter())        

        # ax[4].set_title('Firing Rate from Session '+session_name, fontsize=my_fontsize, color="black")
        ax[4].set_ylabel('sorted units', fontsize=my_fontsize, color="black")
        ax[4].set_xlabel('time (s)', fontsize=my_fontsize, color="black")

        plt.savefig( plot_path+'/'+ 'attention_map_' +str(start_time_bin)+ '_to_'+str(end_time_bin) +'.png' )


        plt.cla()
        plt.clf()
        plt.close()
    
        return


    '''
    def attention_map_2_outputs_with_target_cue_unit_attention(self, session_name, type_name, start_time_bin, end_time_bin, plot_path, my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2, attn_weight_matrix_all, x_target_cue, y_target_cue, firing_rate_collector):

        firing_rate_collector=firing_rate_collector[start_time_bin:end_time_bin,:]
        attn_weight_matrix_all=attn_weight_matrix_all[start_time_bin:end_time_bin,:]

        sns.set(font_scale=3)
        sns.set_style("white")

        plt.rcParams["figure.figsize"] = (30,30)

        time=[]
        for i in range(start_time_bin, end_time_bin):
            time+=[i]

        f, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4,3,2],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05} , constrained_layout=True)

        firing_rate_data = firing_rate_collector.transpose()

        # Eliniate empty units
        valid_rows=[]
        for row_idx in range(firing_rate_data.shape[0]):
            if not np.all( firing_rate_data[row_idx,:] ==0 ):
                valid_rows.append(row_idx)
        firing_rate_data = firing_rate_data[valid_rows,:]

        cbar_kws_firingrate = {"orientation": "horizontal", "shrink": 0.5, "aspect":40,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(firing_rate_data),  4 ]}
        sns.heatmap( data=firing_rate_data , ax=ax[0], vmax=4, cbar_kws=cbar_kws_firingrate, cmap='YlGnBu_r', yticklabels=True, xticklabels=False ) # norm=LogNorm()
    
        ax[0].set_yticklabels(ax[0].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0) # This will get correct row numbers of data matrix

        # b, t = ax[0].get_ylim() # discover the values for bottom and top
        # b += 0.5 # Add 0.5 to the bottom
        # t -= 0.5 # Subtract 0.5 from the top
        # ax[0].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[0].yaxis.set_major_locator(ticker.MultipleLocator(50))
        ax[0].yaxis.set_major_formatter(ticker.ScalarFormatter())

        # ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0) # This will get wrong row numbers of data matrix

        ax[0].set_title('Firing Rate from Session '+session_name, fontsize=my_fontsize, color="black")
        ax[0].set_ylabel('Units', fontsize=my_fontsize, color="black")


        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_all.shape[1]):
            my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )

        attention_map_data = attn_weight_matrix_all.transpose()

        cbar_kws_attention = {"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(attention_map_data), np.max(attention_map_data) ]}
        sns.heatmap( data=attention_map_data , ax=ax[1], cbar_kws=cbar_kws_attention, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False  ) # norm=LogNorm()

        b, t = ax[1].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[1].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[1].yaxis.set_major_locator(ticker.MultipleLocator(50))
        ax[1].yaxis.set_major_formatter(ticker.ScalarFormatter())

        # ax[1].set_yticklabels(ax[1].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0) # This will get wrong row numbers of data matrix
        ax[1].set_title('Attention Map', fontsize=my_fontsize, color="black")
        ax[1].set_ylabel(' Sorted Units ', fontsize=my_fontsize, color="black")

        ax[2].set_title( 'Kinematic Variable Reconstruction', fontsize=my_fontsize, color="black")
        if type_name=='pos':
            ax[2].set_ylabel( 'Position ($mm$)', rotation=90)
        if type_name=='vel':
            ax[2].set_ylabel( 'Velocity ($mm/s$)', rotation=90)
        if type_name=='acc':
            ax[2].set_ylabel( 'Acceleration ($mm/s^2$)', rotation=90)
        ax[2].plot( time, my_prediction_1[start_time_bin:end_time_bin], 'b', linewidth=3, label='x-coor. prediction', alpha=0.8 )
        ax[2].plot( time, Ground_Truth_1[start_time_bin:end_time_bin], 'b--', linewidth=3, label='x-coor. actual', alpha=0.9 )

        ax[2].plot( time, my_prediction_2[start_time_bin:end_time_bin], 'g', linewidth=3, label='y-coor. prediction', alpha=0.8 )
        ax[2].plot( time, Ground_Truth_2[start_time_bin:end_time_bin], 'g--', linewidth=3, label='y-coor. actual', alpha=0.9 )

        # plot target cue change points
        the_x = x_target_cue[start_time_bin:end_time_bin]
        the_y = y_target_cue[start_time_bin:end_time_bin]
        change_points_x = np.where(  np.roll( the_x,1)!= the_x )[0]
        change_points_y = np.where( np.roll( the_y,1)!= the_y )[0]

        change_points_x_set = set(change_points_x)
        change_points_y_set = set(change_points_y)
        
        for ele in list(change_points_x_set.union( change_points_y_set )):
            ax[2].axvline( time[0]+ele , color='black' , linewidth=5, alpha=0.3 )
        
        # this is for target cue checking
        # ax[2].plot(time, x_target_cue[start_time_bin:end_time_bin] , 'ob', linewidth=3, label='x-coor. cue', alpha=0.8)
        # ax[2].plot(time, y_target_cue[start_time_bin:end_time_bin], 'og',linewidth=3, label='y-coor. cue', alpha=0.8)

        ax[2].legend(loc='upper right', fontsize=my_fontsize*0.8)

        ax[2].set_xlim([ time[0], time[-1] ])
        ax[2].set_xlabel('Time Bins', fontsize=my_fontsize, color="black")

        plt.savefig( plot_path+'/'+ 'attention_map_' +str(start_time_bin)+ '_to_'+str(end_time_bin) +'.png' )

        plt.cla()
        plt.clf()
        plt.close()

        

        return
    '''

    ''' Obsolete

    def attention_map_2_outputs_bidir_sep( self, session_name, type_name, start_time_bin, end_time_bin, plot_path, my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2, attn_weight_matrix_all_forward, attn_weight_matrix_all_backward, firing_rate_collector):
        firing_rate_collector = firing_rate_collector[start_time_bin:end_time_bin,:]
        attn_weight_matrix_all_forward = attn_weight_matrix_all_forward[start_time_bin:end_time_bin,:]
        attn_weight_matrix_all_backward = attn_weight_matrix_all_backward[start_time_bin:end_time_bin,:]

        sns.set(font_scale=3)
        plt.rcParams["figure.figsize"] = (30,40)

        time=[]
        for i in range(start_time_bin, end_time_bin):
            time+=[i]

        f, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [4,3,3,2],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05} , constrained_layout=True)


        firing_rate_data = firing_rate_collector.transpose()

        # Eliniate empty units
        valid_rows=[]
        for row_idx in range(firing_rate_data.shape[0]):
            if not np.all( firing_rate_data[row_idx,:] ==0 ):
                valid_rows.append(row_idx)
        firing_rate_data = firing_rate_data[valid_rows,:]

        cbar_kws_firingrate = {"orientation": "horizontal", "shrink": 0.5, "aspect":40,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(firing_rate_data),  4 ]}
        sns.heatmap( data=firing_rate_data , ax=ax[0], vmax=4, cbar_kws=cbar_kws_firingrate, cmap='YlGnBu_r', yticklabels=True, xticklabels=False ) # norm=LogNorm()
    

        ax[0].set_yticklabels(ax[0].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0) # This will get correct row numbers of data matrix

        ax[0].yaxis.set_major_locator(ticker.MultipleLocator(50))
        ax[0].yaxis.set_major_formatter(ticker.ScalarFormatter())

        ax[0].set_title('Firing Rate from Session '+session_name, fontsize=my_fontsize, color="black")
        ax[0].set_ylabel('Units', fontsize=my_fontsize, color="black")


        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_all_forward.shape[1]):
            my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all_forward.shape[1]) -1 - ticks_label) )

        attention_map_data = attn_weight_matrix_all_forward.transpose()

        cbar_kws_attention = {"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(attention_map_data), np.max(attention_map_data) ]}
        sns.heatmap( data=attention_map_data , ax=ax[1], cbar_kws=cbar_kws_attention, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False ) # norm=LogNorm()

        b, t = ax[1].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[1].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[1].set_yticklabels(ax[1].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0)
        ax[1].set_title('Attention Map Foward', fontsize=my_fontsize, color="black")
        ax[1].set_ylabel('Past Time Bins', fontsize=my_fontsize, color="black")



        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_all_forward.shape[1]):
            my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all_backward.shape[1]) -1 - ticks_label) )

        attention_map_data = attn_weight_matrix_all_backward.transpose()

        cbar_kws_attention = {"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(attention_map_data), np.max(attention_map_data) ]}
        sns.heatmap( data=attention_map_data , ax=ax[2], cbar_kws=cbar_kws_attention, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False ) # norm=LogNorm()

        b, t = ax[2].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[2].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[2].set_yticklabels(ax[2].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0)
        ax[2].set_title('Attention Map Backward', fontsize=my_fontsize, color="black")
        ax[2].set_ylabel('Past Time Bins', fontsize=my_fontsize, color="black")


        ax[3].set_title( 'Kinematic Variable Reconstruction', fontsize=my_fontsize, color="black")
        if type_name=='pos':
            ax[3].set_ylabel( 'Position (mm)', rotation=90)
        if type_name=='vel':
            ax[3].set_ylabel( 'Velocity (mm/s)', rotation=90)
        if type_name=='acc':
            ax[3].set_ylabel( 'Acceleration (mm/$s^2$)', rotation=90)
        ax[3].plot( time, my_prediction_1[start_time_bin:end_time_bin], 'b', linewidth=3, label='x-coor. prediction', alpha=0.7 )
        ax[3].plot( time, Ground_Truth_1[start_time_bin:end_time_bin], 'b--', linewidth=3, label='x-coor. actual', alpha=0.8 )

        ax[3].plot( time, my_prediction_2[start_time_bin:end_time_bin], 'g', linewidth=3, label='y-coor. prediction', alpha=0.7 )
        ax[3].plot( time, Ground_Truth_2[start_time_bin:end_time_bin], 'g--', linewidth=3, label='y-coor. actual', alpha=0.8 )
    
        ax[3].legend(loc='upper right', fontsize=my_fontsize*0.8)

        ax[3].set_xlim([ time[0], time[-1] ])
        ax[3].set_xlabel('Time Bins', fontsize=my_fontsize, color="black")

        plt.savefig( plot_path+'/'+ 'attention_map_' +str(start_time_bin)+ '_to_'+str(end_time_bin) +'.png' )

        plt.cla()
        plt.clf()
        plt.close()
        return

    def attention_map_2_outputs_bidir_sep_with_target_cue(self, session_name, type_name, start_time_bin, end_time_bin, plot_path, my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2, attn_weight_matrix_all_forward, attn_weight_matrix_all_backward, x_target_cue, y_target_cue, firing_rate_collector):
        firing_rate_collector = firing_rate_collector[start_time_bin:end_time_bin,:]
        attn_weight_matrix_all_forward = attn_weight_matrix_all_forward[start_time_bin:end_time_bin,:]
        attn_weight_matrix_all_backward = attn_weight_matrix_all_backward[start_time_bin:end_time_bin,:]

        sns.set(font_scale=3)
        plt.rcParams["figure.figsize"] = (30,40)

        time=[]
        for i in range(start_time_bin, end_time_bin):
            time+=[i]

        f, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [4,3,3,2],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05} , constrained_layout=True)


        firing_rate_data = firing_rate_collector.transpose()

        # Eliniate empty units
        valid_rows=[]
        for row_idx in range(firing_rate_data.shape[0]):
            if not np.all( firing_rate_data[row_idx,:] ==0 ):
                valid_rows.append(row_idx)
        firing_rate_data = firing_rate_data[valid_rows,:]

        cbar_kws_firingrate = {"orientation": "horizontal", "shrink": 0.5, "aspect":40,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(firing_rate_data),  4 ]}
        sns.heatmap( data=firing_rate_data , ax=ax[0], vmax=4, cbar_kws=cbar_kws_firingrate, cmap='YlGnBu_r', yticklabels=True, xticklabels=False ) # norm=LogNorm()
    

        ax[0].set_yticklabels(ax[0].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0) # This will get correct row numbers of data matrix

        ax[0].yaxis.set_major_locator(ticker.MultipleLocator(50))
        ax[0].yaxis.set_major_formatter(ticker.ScalarFormatter())

        ax[0].set_title('Firing Rate from Session '+session_name, fontsize=my_fontsize, color="black")
        ax[0].set_ylabel('Units', fontsize=my_fontsize, color="black")


        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_all_forward.shape[1]):
            my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all_forward.shape[1]) -1 - ticks_label) )

        attention_map_data = attn_weight_matrix_all_forward.transpose()

        cbar_kws_attention = {"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(attention_map_data), np.max(attention_map_data) ]}
        sns.heatmap( data=attention_map_data , ax=ax[1], cbar_kws=cbar_kws_attention, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False ) # norm=LogNorm()

        b, t = ax[1].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[1].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[1].set_yticklabels(ax[1].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0)
        ax[1].set_title('Attention Map Foward', fontsize=my_fontsize, color="black")
        ax[1].set_ylabel('Past Time Bins', fontsize=my_fontsize, color="black")



        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_all_forward.shape[1]):
            my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all_backward.shape[1]) -1 - ticks_label) )

        attention_map_data = attn_weight_matrix_all_backward.transpose()

        cbar_kws_attention = {"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(attention_map_data), np.max(attention_map_data) ]}
        sns.heatmap( data=attention_map_data , ax=ax[2], cbar_kws=cbar_kws_attention, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False ) # norm=LogNorm()

        b, t = ax[2].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[2].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[2].set_yticklabels(ax[2].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0)
        ax[2].set_title('Attention Map Backward', fontsize=my_fontsize, color="black")
        ax[2].set_ylabel('Past Time Bins', fontsize=my_fontsize, color="black")


        ax[3].set_title( 'Kinematic Variable Reconstruction', fontsize=my_fontsize, color="black")
        if type_name=='pos':
            ax[3].set_ylabel( 'Position (mm)', rotation=90)
        if type_name=='vel':
            ax[3].set_ylabel( 'Velocity (mm/s)', rotation=90)
        if type_name=='acc':
            ax[3].set_ylabel( 'Acceleration (mm/$s^2$)', rotation=90)
        ax[3].plot( time, my_prediction_1[start_time_bin:end_time_bin], 'b', linewidth=3, label='x-coor. prediction', alpha=0.7 )
        ax[3].plot( time, Ground_Truth_1[start_time_bin:end_time_bin], 'b--', linewidth=3, label='x-coor. actual', alpha=0.8 )

        ax[3].plot( time, my_prediction_2[start_time_bin:end_time_bin], 'g', linewidth=3, label='y-coor. prediction', alpha=0.7 )
        ax[3].plot( time, Ground_Truth_2[start_time_bin:end_time_bin], 'g--', linewidth=3, label='y-coor. actual', alpha=0.8 )

        # plot target cue change points
        the_x = x_target_cue[start_time_bin:end_time_bin]
        the_y = y_target_cue[start_time_bin:end_time_bin]
        change_points_x = np.where(  np.roll( the_x,1)!= the_x )[0]
        change_points_y = np.where( np.roll( the_y,1)!= the_y )[0]

        change_points_x_set = set(change_points_x)
        change_points_y_set = set(change_points_y)
        
        for ele in list(change_points_x_set.union( change_points_y_set )):
            ax[3].axvline( time[0]+ele , color='black' , linewidth=5, alpha=0.3 )
        
        # this is for target cue checking
        # ax[3].plot(time, x_target_cue[start_time_bin:end_time_bin] , 'ob', linewidth=3, label='x-coor. cue', alpha=0.8)
        # ax[3].plot(time, y_target_cue[start_time_bin:end_time_bin], 'og',linewidth=3, label='y-coor. cue', alpha=0.8)

        ax[3].legend(loc='upper right', fontsize=my_fontsize*0.8)

        ax[3].set_xlim([ time[0], time[-1] ])
        ax[3].set_xlabel('Time Bins', fontsize=my_fontsize, color="black")

        plt.savefig( plot_path+'/'+ 'attention_map_' +str(start_time_bin)+ '_to_'+str(end_time_bin) +'.png' )

        plt.cla()
        plt.clf()
        plt.close()
        return


    def attention_map_2_outputs_2_stream_with_target_cue(self, session_name, type_name, start_time_bin, end_time_bin, plot_path, my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2, attn_weight_matrix_forward_M1, attn_weight_matrix_forward_S1, x_target_cue, y_target_cue, firing_rate_collector):
        firing_rate_collector = firing_rate_collector[start_time_bin:end_time_bin,:]
        attn_weight_matrix_forward_M1 = attn_weight_matrix_forward_M1[start_time_bin:end_time_bin,:]
        attn_weight_matrix_forward_S1 = attn_weight_matrix_forward_S1[start_time_bin:end_time_bin,:]

        sns.set(font_scale=3)
        plt.rcParams["figure.figsize"] = (30,40)

        time=[]
        for i in range(start_time_bin, end_time_bin):
            time+=[i]

        f, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [4,3,3,2],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05} , constrained_layout=True)


        firing_rate_data = firing_rate_collector.transpose()

        # Eliniate empty units
        valid_rows=[]
        for row_idx in range(firing_rate_data.shape[0]):
            if not np.all( firing_rate_data[row_idx,:] ==0 ):
                valid_rows.append(row_idx)
        firing_rate_data = firing_rate_data[valid_rows,:]

        cbar_kws_firingrate = {"orientation": "horizontal", "shrink": 0.5, "aspect":40,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(firing_rate_data),  4 ]}
        sns.heatmap( data=firing_rate_data , ax=ax[0], vmax=4, cbar_kws=cbar_kws_firingrate, cmap='YlGnBu_r', yticklabels=True, xticklabels=False ) # norm=LogNorm()
    

        ax[0].set_yticklabels(ax[0].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0) # This will get correct row numbers of data matrix

        ax[0].yaxis.set_major_locator(ticker.MultipleLocator(50))
        ax[0].yaxis.set_major_formatter(ticker.ScalarFormatter())

        ax[0].set_title('Firing Rate from Session '+session_name, fontsize=my_fontsize, color="black")
        ax[0].set_ylabel('Units', fontsize=my_fontsize, color="black")


        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_forward_M1.shape[1]):
            my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_forward_M1.shape[1]) -1 - ticks_label) )

        attention_map_data = attn_weight_matrix_forward_M1.transpose()

        cbar_kws_attention = {"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(attention_map_data), np.max(attention_map_data) ]}
        sns.heatmap( data=attention_map_data , ax=ax[1], cbar_kws=cbar_kws_attention, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False ) # norm=LogNorm()

        b, t = ax[1].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[1].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[1].set_yticklabels(ax[1].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0)
        ax[1].set_title('Attention Map M1', fontsize=my_fontsize, color="black")
        ax[1].set_ylabel('Past Time Bins', fontsize=my_fontsize, color="black")



        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_forward_S1.shape[1]):
            my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_forward_S1.shape[1]) -1 - ticks_label) )

        attention_map_data = attn_weight_matrix_forward_S1.transpose()

        cbar_kws_attention = {"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(attention_map_data), np.max(attention_map_data) ]}
        sns.heatmap( data=attention_map_data , ax=ax[2], cbar_kws=cbar_kws_attention, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False ) # norm=LogNorm()

        b, t = ax[2].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[2].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[2].set_yticklabels(ax[2].get_ymajorticklabels(), fontsize = my_fontsize, rotation=0)
        ax[2].set_title('Attention Map S1', fontsize=my_fontsize, color="black")
        ax[2].set_ylabel('Past Time Bins', fontsize=my_fontsize, color="black")


        ax[3].set_title( 'Kinematic Variable Reconstruction', fontsize=my_fontsize, color="black")
        if type_name=='pos':
            ax[3].set_ylabel( 'Position (mm)', rotation=90)
        if type_name=='vel':
            ax[3].set_ylabel( 'Velocity (mm/s)', rotation=90)
        if type_name=='acc':
            ax[3].set_ylabel( 'Acceleration (mm/$s^2$)', rotation=90)
        ax[3].plot( time, my_prediction_1[start_time_bin:end_time_bin], 'b', linewidth=3, label='x-coor. prediction', alpha=0.7 )
        ax[3].plot( time, Ground_Truth_1[start_time_bin:end_time_bin], 'b--', linewidth=3, label='x-coor. actual', alpha=0.8 )

        ax[3].plot( time, my_prediction_2[start_time_bin:end_time_bin], 'g', linewidth=3, label='y-coor. prediction', alpha=0.7 )
        ax[3].plot( time, Ground_Truth_2[start_time_bin:end_time_bin], 'g--', linewidth=3, label='y-coor. actual', alpha=0.8 )

        # plot target cue change points
        the_x = x_target_cue[start_time_bin:end_time_bin]
        the_y = y_target_cue[start_time_bin:end_time_bin]
        change_points_x = np.where(  np.roll( the_x,1)!= the_x )[0]
        change_points_y = np.where( np.roll( the_y,1)!= the_y )[0]

        change_points_x_set = set(change_points_x)
        change_points_y_set = set(change_points_y)
        
        for ele in list(change_points_x_set.union( change_points_y_set )):
            ax[3].axvline( time[0]+ele , color='black' , linewidth=5, alpha=0.3 )
        
        # this is for target cue checking
        # ax[3].plot(time, x_target_cue[start_time_bin:end_time_bin] , 'ob', linewidth=3, label='x-coor. cue', alpha=0.8)
        # ax[3].plot(time, y_target_cue[start_time_bin:end_time_bin], 'og',linewidth=3, label='y-coor. cue', alpha=0.8)

        ax[3].legend(loc='upper right', fontsize=my_fontsize*0.8)

        ax[3].set_xlim([ time[0], time[-1] ])
        ax[3].set_xlabel('Time Bins', fontsize=my_fontsize, color="black")

        plt.savefig( plot_path+'/'+ 'attention_map_' +str(start_time_bin)+ '_to_'+str(end_time_bin) +'.png' )

        plt.cla()
        plt.clf()
        plt.close()
        return

    def hidden_state_bar_plot( self, session_name,  plot_path, hidden_state_1, hidden_state_2):

        plt.rcParams["figure.figsize"] = (16,9)
        plt.rcParams['xtick.labelsize'] = my_fontsize*0.5
        plt.rcParams['ytick.labelsize'] = my_fontsize*0.5

        f, ax = plt.subplots(2, 1, gridspec_kw={ 'height_ratios': [1,1], "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.9, "bottom":0.1} , constrained_layout=True)

        bins = [ i for i in np.arange( -2, 2, 0.01 )]

        ax[0].hist( x=hidden_state_1[0,:] , bins=bins )
        ax[0].set_title(' Layer 1 Hidden State Distribution', fontsize=my_fontsize*0.5 )

        ax[1].hist( x=hidden_state_2[0,:] , bins=bins )
        ax[1].set_title(' Layer 2 Hidden State Distribution', fontsize=my_fontsize*0.5 )


        plt.savefig( plot_path+'/'+ 'hidden_state_distribution'+'.png' )

        plt.cla()
        plt.clf()
        plt.close()
        return

    '''