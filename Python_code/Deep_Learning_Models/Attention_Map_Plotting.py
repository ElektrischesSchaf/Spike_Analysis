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

my_fontsize=30

class Plotting():

    # for 1-output GRU model
    def attention_map( self, end_time_bin, plot_path, my_prediction, Ground_Truth, attn_weight_matrix_all ):
        attn_weight_matrix_all=attn_weight_matrix_all[:end_time_bin,:]

        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (16,9)

        # grid_kws = {"height_ratios": (.3, .02, .3, .3), "hspace": 0.01}
        time=[]
        for i in range(end_time_bin):
            time+=[i]

        f, (ax, ax2) = plt.subplots(2)

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
    def attention_map_2_outputs( self, session_name, type_name, start_time_bin, end_time_bin, plot_path, my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2, attn_weight_matrix_all, firing_rate_collector):

        firing_rate_collector=firing_rate_collector[start_time_bin:end_time_bin,:]
        attn_weight_matrix_all=attn_weight_matrix_all[start_time_bin:end_time_bin,:]

        sns.set(font_scale=3)
        plt.rcParams["figure.figsize"] = (30,30)

        time=[]
        for i in range(start_time_bin, end_time_bin):
            time+=[i]

        f, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4,3,2],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05} )

        firing_rate_data = firing_rate_collector.transpose()
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
            ax[2].set_ylabel( 'Acceleration (mm/s^2)', rotation=90)
        ax[2].plot( time, my_prediction_1[start_time_bin:end_time_bin], 'b', linewidth=3, label='x-axis prediction', alpha=0.7 )
        ax[2].plot( time, Ground_Truth_1[start_time_bin:end_time_bin], 'b--', linewidth=3, label='x-axis actual', alpha=0.8 )

        ax[2].plot( time, my_prediction_2[start_time_bin:end_time_bin], 'g', linewidth=3, label='y-axis prediction', alpha=0.7 )
        ax[2].plot( time, Ground_Truth_2[start_time_bin:end_time_bin], 'g--', linewidth=3, label='y-axis actual', alpha=0.8 )
    
        ax[2].legend(loc='upper right', fontsize=my_fontsize*0.8)

        ax[2].set_xlim([ time[0], time[-1] ])
        ax[2].set_xlabel('Time Bins', fontsize=my_fontsize, color="black")

        plt.savefig( plot_path+'/'+ 'attention_map_' +str(start_time_bin)+ '_to_'+str(end_time_bin) +'.png' )

        plt.cla()
        plt.clf()
        plt.close()

        return

    # for 2-output GRU model
    def attention_map_2_outputs_with_target_cue( self, session_name, type_name, start_time_bin, end_time_bin, plot_path, my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2, attn_weight_matrix_all, x_target_cue, y_target_cue, firing_rate_collector):

        firing_rate_collector=firing_rate_collector[start_time_bin:end_time_bin,:]
        attn_weight_matrix_all=attn_weight_matrix_all[start_time_bin:end_time_bin,:]

        sns.set(font_scale=3)
        plt.rcParams["figure.figsize"] = (30,30)

        time=[]
        for i in range(start_time_bin, end_time_bin):
            time+=[i]

        f, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4,3,2],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05} )

        firing_rate_data = firing_rate_collector.transpose()
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
            ax[2].set_ylabel( 'Acceleration (mm/s^2)', rotation=90)
        ax[2].plot( time, my_prediction_1[start_time_bin:end_time_bin], 'b', linewidth=3, label='x-axis prediction', alpha=0.7 )
        ax[2].plot( time, Ground_Truth_1[start_time_bin:end_time_bin], 'b--', linewidth=3, label='x-axis actual', alpha=0.8 )

        ax[2].plot( time, my_prediction_2[start_time_bin:end_time_bin], 'g', linewidth=3, label='y-axis prediction', alpha=0.7 )
        ax[2].plot( time, Ground_Truth_2[start_time_bin:end_time_bin], 'g--', linewidth=3, label='y-axis actual', alpha=0.8 )

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
        # ax[2].plot(time, x_target_cue[start_time_bin:end_time_bin] , 'ob', linewidth=3, label='x-axis cue', alpha=0.8)
        # ax[2].plot(time, y_target_cue[start_time_bin:end_time_bin], 'og',linewidth=3, label='y-axis cue', alpha=0.8)

        ax[2].legend(loc='upper right', fontsize=my_fontsize*0.8)

        ax[2].set_xlim([ time[0], time[-1] ])
        ax[2].set_xlabel('Time Bins', fontsize=my_fontsize, color="black")

        plt.savefig( plot_path+'/'+ 'attention_map_' +str(start_time_bin)+ '_to_'+str(end_time_bin) +'.png' )

        plt.cla()
        plt.clf()
        plt.close()

        return

    def attention_map_2_outputs_bidir_sep( self, session_name, type_name, start_time_bin, end_time_bin, plot_path, my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2, attn_weight_matrix_all_forward, attn_weight_matrix_all_backward, firing_rate_collector):
        firing_rate_collector = firing_rate_collector[start_time_bin:end_time_bin,:]
        attn_weight_matrix_all_forward = attn_weight_matrix_all_forward[start_time_bin:end_time_bin,:]
        attn_weight_matrix_all_backward = attn_weight_matrix_all_backward[start_time_bin:end_time_bin,:]

        sns.set(font_scale=3)
        plt.rcParams["figure.figsize"] = (30,40)

        time=[]
        for i in range(start_time_bin, end_time_bin):
            time+=[i]

        f, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [4,3,3,2],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05} )


        firing_rate_data = firing_rate_collector.transpose()
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
            ax[3].set_ylabel( 'Acceleration (mm/s^2)', rotation=90)
        ax[3].plot( time, my_prediction_1[start_time_bin:end_time_bin], 'b', linewidth=3, label='x-axis prediction', alpha=0.7 )
        ax[3].plot( time, Ground_Truth_1[start_time_bin:end_time_bin], 'b--', linewidth=3, label='x-axis actual', alpha=0.8 )

        ax[3].plot( time, my_prediction_2[start_time_bin:end_time_bin], 'g', linewidth=3, label='y-axis prediction', alpha=0.7 )
        ax[3].plot( time, Ground_Truth_2[start_time_bin:end_time_bin], 'g--', linewidth=3, label='y-axis actual', alpha=0.8 )
    
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

        f, ax = plt.subplots(2, 1, gridspec_kw={ 'height_ratios': [1,1], "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.9, "bottom":0.1} )

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