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

my_fontsize=30

class Plotting():

    # for 1-output GRU model
    def attention_map( self, time_bin_to_plot, plot_path, my_prediction, Ground_Truth, attn_weight_matrix_all ):
        attn_weight_matrix_all=attn_weight_matrix_all[:time_bin_to_plot,:]

        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (16,9)

        # grid_kws = {"height_ratios": (.3, .02, .3, .3), "hspace": 0.01}
        time=[]
        for i in range(time_bin_to_plot):
            time+=[i]

        f, (ax, ax2) = plt.subplots(2)

        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_all.shape[1]):
            my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )

        ax = sns.heatmap( attn_weight_matrix_all.transpose(), ax=ax, cbar=False, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False,  norm=LogNorm())
        plt.yticks(rotation=90)
        ax.set_title('Attention Map')
        ax.set_ylabel('Past Time Bins')

        ax2 = plt.plot( time, my_prediction[:time_bin_to_plot], 'b', linewidth=3, label='Prediction' )
        ax2 = plt.plot( time, Ground_Truth[:time_bin_to_plot], 'r', linewidth=3, label='Actual', alpha=0.7 )
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
    def attention_map_2_outputs( self, time_bin_to_plot, plot_path, my_prediction_1, Ground_Truth_1, my_prediction_2, Ground_Truth_2, attn_weight_matrix_all, firing_rate_collector):

        firing_rate_collector=firing_rate_collector[:time_bin_to_plot,:]
        attn_weight_matrix_all=attn_weight_matrix_all[:time_bin_to_plot,:]

        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (16,9)

        time=[]
        for i in range(time_bin_to_plot):
            time+=[i]

        f, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3,2],  "hspace":0.2 ,"left":0.1, "right":0.9, "top":0.95, "bottom":0.05} )

        my_yticklabels=[]
        for ticks_label in range(attn_weight_matrix_all.shape[1]):
            my_yticklabels.append( 't-' + str(  int(attn_weight_matrix_all.shape[1]) -1 - ticks_label) )

        attention_map_data=attn_weight_matrix_all.transpose()
        cbar_kws={"orientation": "horizontal", "shrink": 0.5, "aspect":50,"use_gridspec":"True", "fraction":0.01 , "pad":0.03, 'ticks' : [ np.min(attention_map_data), np.max(attention_map_data) ]}
        sns.heatmap( data=attention_map_data , ax=ax[0], cbar_kws=cbar_kws, cmap='coolwarm', yticklabels=my_yticklabels, xticklabels=False ) # norm=LogNorm()

        b, t = ax[0].get_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        ax[0].set_ylim(b, t) # update the ylim(bottom, top) values

        ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
        ax[0].set_title('Attention Map')
        ax[0].set_ylabel('Past Time Bins')

        ax[1].set_title( 'Kinematic Variable Reconstruction')
        ax[1].set_ylabel( 'Position (mm)', rotation=90)
        # ax[1].set_ylabel( 'Velocity (mm/s)', rotation=90)
        ax[1].plot( time, my_prediction_1[:time_bin_to_plot], 'b', linewidth=3, label='x-axis prediction', alpha=0.7 )
        ax[1].plot( time, Ground_Truth_1[:time_bin_to_plot], 'b--', linewidth=3, label='x-axis actual', alpha=0.8 )

        ax[1].plot( time, my_prediction_2[:time_bin_to_plot], 'g', linewidth=3, label='y-axis prediction', alpha=0.7 )
        ax[1].plot( time, Ground_Truth_2[:time_bin_to_plot], 'g--', linewidth=3, label='y-axis actual', alpha=0.8 )
    
        ax[1].legend(loc='upper right', fontsize=my_fontsize*0.3)

        ax[1].set_xlim([ time[0], time[-1] ])
        ax[1].set_xlabel('Time Bins')

        plt.savefig( plot_path+'/'+ 'attention_map' +'.png' )
        plt.cla()
        plt.clf()
        plt.close()

        return