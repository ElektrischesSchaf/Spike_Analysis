import seaborn as sns
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
file_path= './new/'
file_list= os.listdir(file_path)
print(file_list)
for i in range(len(file_path)):
    file_name=str(file_list[i])[:-4]
    df=pd.read_csv( file_path+file_name+'.csv', sep=',',header=None )
    rows=df.values
    print(rows.shape)
    # plt.figure(1, figsize=(16, 9) )
    # ax = sns.heatmap(rows, vmin=0, vmax=1)
    sns.set(font_scale=1.4)
    plt.rcParams["figure.figsize"] = (16,9)

    grid_kws = {"height_ratios": (.9, .02), "hspace": 0.3}

    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    plt.title(file_name, fontsize=25)
    ax = sns.heatmap(rows, ax=ax,
                    cbar_ax=cbar_ax,
                    cmap="YlGnBu",
                    yticklabels=False,
                    cbar_kws={"orientation": "horizontal"})

    # plt.tight_layout()
    # plt.show()
    plt.savefig( './yee/'+file_name+'.png' )