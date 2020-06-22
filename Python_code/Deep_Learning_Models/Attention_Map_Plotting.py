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

class Plotting():

    def attention_map( self, time_bin_to_plot, plot_path, my_prediction, Ground_Truth, attn_weight_matrix_all ):
        attn_weight_matrix_all=attn_weight_matrix_all[:time_bin_to_plot,:]

        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (16,9)

        # grid_kws = {"height_ratios": (.3, .02, .3, .3), "hspace": 0.01}
        time=[]
        for i in range(time_bin_to_plot):
            time+=[i]

        f, (ax, ax2) = plt.subplots(2)

        ax = sns.heatmap( attn_weight_matrix_all.transpose(), ax=ax, cbar=False, cmap='YlGnBu', yticklabels=False, xticklabels=False)
        ax.set_title('Attention Map')
        ax.set_ylabel('Past Time Bins')

        ax2 = plt.plot( time, my_prediction[:time_bin_to_plot], 'b', linewidth=5, label='Prediction' )
        ax2 = plt.plot( time, Ground_Truth[:time_bin_to_plot], 'r', linewidth=5, label='Actual', alpha=0.7 )
        plt.legend(loc='upper right', fontsize=10)

        plt.xlim([ time[0], time[-1] ])
        plt.xlabel('Time Bin', fontsize=25)
        plt.tight_layout()
        plt.savefig( plot_path+'/'+ 'attention_map' +'.png' )
        plt.cla()
        plt.clf()
        plt.close()

        return