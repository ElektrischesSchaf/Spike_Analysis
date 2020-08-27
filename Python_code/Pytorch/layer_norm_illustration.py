# https://medium.com/@omkar.nallagoni/activation-functions-with-derivative-and-python-code-sigmoid-vs-tanh-vs-relu-44d23915c1f4

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
# Figures
import imageio
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)


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
import shutil

CWD_origin=os.getcwd()

layer_norm_illustration = os.path.join(CWD_origin, 'layer_norm_illustration')
if not os.path.exists(layer_norm_illustration):
    os.mkdir(layer_norm_illustration)


import seaborn as sns

my_fontsize = 30
my_width = 16
my_height = 9

def sigmoid(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)  
    return s,ds

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return t,dt



z=np.arange(-6,6,0.01)
tanh(z)[0].size,tanh(z)[1].size
sigmoid(z)

# Setup centered axes
fig, ax = plt.subplots(figsize=(my_width, my_height))
my_fontsize = 30
plt.rcParams['xtick.labelsize'] = my_fontsize
plt.rcParams['ytick.labelsize'] = my_fontsize
ax.set_xticks([-6,-4,-2,0,2,4,6])
ax.set_yticks([-1,1])

ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Create and show plot
ax.plot(z,tanh(z)[0], color='blue', linewidth=3, label="tanh(x)")
# ax.plot(z,tanh(z)[1], color="#9621E2", linewidth=3, label="derivative")

ax.plot(z,sigmoid(z)[0], color='green', linewidth=3, label="sigmoid(x)")
# ax.plot(z,sigmoid(z)[1], color="#9621E2", linewidth=3, label="derivative")

ax.legend(loc="upper left", frameon=True, fontsize=my_fontsize)


plt.tight_layout()
plt.savefig(layer_norm_illustration+'/'+'sigmoid_and_tanh.png')

plt.cla()
plt.clf()
fig.clear()


# Setup centered axes
fig, ax = plt.subplots(figsize=(my_width, my_height))

plt.rcParams['xtick.labelsize'] = my_fontsize
plt.rcParams['ytick.labelsize'] = my_fontsize
ax.set_xticks([-6,-4,-2,0,2,4,6])
ax.set_yticks([1])
ax.set_ylim([0,1.1])

ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Create and show plot

ax.plot(z,tanh(z)[1], color="blue", linewidth=3, label=r'$\frac{d}{dx}$ tanh(x)')

ax.plot(z,sigmoid(z)[1], color="green", linewidth=3, label=r'$\frac{d}{dx}$ sigmoid(x)')

ax.legend(loc="upper left", frameon=True, fontsize=my_fontsize)



plt.tight_layout()
plt.savefig(layer_norm_illustration+'/'+'derivative_sigmoid_and_tanh.png')


plt.cla()
plt.clf()
fig.clear()


# torch.normal( mean, std, [size] )
input_1 = torch.normal( 7, 1, [1,256] )
input_2 = torch.normal( 0, 10, [1,256])

m_1 = torch.nn.LayerNorm(input_1.size(), elementwise_affine=False)
m_2 = torch.nn.LayerNorm(input_2.size(), elementwise_affine=False)

output_1 = m_1(input_1)
output_2 = m_2(input_2)

input_1=input_1.numpy()
output_1=output_1.numpy()
input_2=input_2.numpy()
output_2=output_2.numpy()

bins = [i for i in np.arange( -10, 10, 0.2 )]

fig=plt.figure(figsize=(my_width,my_height))
plt.rcParams['xtick.labelsize'] = my_fontsize
plt.rcParams['ytick.labelsize'] = my_fontsize
plt.hist(input_1[0,:], bins=bins, color='blue', alpha=0.5, label='Mean:7, STD:1')
plt.hist(input_2[0,:], bins=bins, color='green', alpha=0.5,label='Mean:0, STD:10')

plt.xlabel('x-axis', fontsize=my_fontsize)
plt.xticks([-10,-8,-6,-4,-2,0,2,4,6,8,10])
plt.yticks([])
plt.legend(loc="upper left", frameon=True, fontsize=my_fontsize)
plt.tight_layout()
plt.savefig( layer_norm_illustration+'/'+ 'activation_histogram_before.png')


plt.cla()
plt.clf()
fig.clear()

fig=plt.figure(figsize=(my_width,my_height))
plt.rcParams['xtick.labelsize'] = my_fontsize
plt.rcParams['ytick.labelsize'] = my_fontsize
plt.hist(output_1[0,:], bins=bins, color='blue', alpha=0.5, label='Mean: 0, STD: 1')
plt.hist(output_2[0,:], bins=bins, color='green', alpha=0.5,label='Mean: 0, STD: 1')

plt.xlabel('x-axis', fontsize=my_fontsize)
plt.xticks([-10,-8,-6,-4,-2,0,2,4,6,8,10])
plt.yticks([])
plt.legend(loc="upper left", frameon=True, fontsize=my_fontsize)
plt.tight_layout()
plt.savefig( layer_norm_illustration+'/'+ 'activation_histogram_after.png')

plt.cla()
plt.clf()
fig.clear()