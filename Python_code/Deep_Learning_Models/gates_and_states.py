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

class compute():
    def comp_and_save(self, input, hidden_State, cell_state, feature_name, seq_length, save_path, w_ii, w_if, w_ic, w_io, w_hi, w_hf, w_hc, w_ho, b_ii, b_if, b_ic, b_io, b_hi, b_hf, b_hc, b_ho):
        x=input
        yee=seq_length
        info_path=save_path
        feature_name=feature_name
        hidden_State=hidden_State
        cell_state=cell_state


        hidden_state=hidden_State.cpu().data.numpy()
        hidden_state=np.transpose(hidden_state)
        df=pd.DataFrame(hidden_state)
        df.to_csv(os.path.join(info_path, feature_name+'_hidden_state.csv'), index=False, header=False)

        if cell_state!=None:
            cell_state=cell_state.cpu().data.numpy()
            cell_state=np.transpose(cell_state)
            df=pd.DataFrame(cell_state)
            df.to_csv(os.path.join(info_path, feature_name+'_cell_state.csv'), index=False, header=False)

        forget_gate=[]                    
        for index_seq in range( yee ):
            yee_hidden=np.matmul(w_hf.cpu().data.numpy(), hidden_state[:,index_seq]) + b_hf.cpu().data.numpy()
            x_temp=np.transpose( x[index_seq,:] )
            yee_input=np.matmul( w_if.cpu().data.numpy(), x_temp) + b_if.cpu().data.numpy()
            forget_gate+=[   torch.sigmoid(    torch.from_numpy(yee_hidden)+torch.from_numpy(yee_input)   ).numpy()  ]
        forget_gate=np.transpose(forget_gate)
        df=pd.DataFrame(forget_gate)
        df.to_csv(os.path.join(info_path, feature_name+'_forget_gate.csv'), index=False, header=False)

        input_gate=[]                    
        for index_seq in range( yee ):
            yee_hidden=np.matmul(w_hi.cpu().data.numpy(), hidden_state[:,index_seq]) + b_hi.cpu().data.numpy()
            x_temp=np.transpose( x[index_seq,:] )
            yee_input=np.matmul( w_ii.cpu().data.numpy(), x_temp) + b_ii.cpu().data.numpy()
            input_gate+=[   torch.sigmoid(    torch.from_numpy(yee_hidden)+torch.from_numpy(yee_input)   ).numpy()  ]
        input_gate=np.transpose(input_gate)
        df=pd.DataFrame(input_gate)
        df.to_csv(os.path.join(info_path, feature_name+'_input_gate.csv'), index=False, header=False)

        output_gate=[]                    
        for index_seq in range( yee ):
            yee_hidden=np.matmul(w_ho.cpu().data.numpy(), hidden_state[:,index_seq]) + b_ho.cpu().data.numpy()
            x_temp=np.transpose( x[index_seq,:] )
            yee_input=np.matmul( w_io.cpu().data.numpy(), x_temp) + b_io.cpu().data.numpy()
            output_gate+=[   torch.sigmoid(    torch.from_numpy(yee_hidden)+torch.from_numpy(yee_input)   ).numpy()  ]
        output_gate=np.transpose(output_gate)
        df=pd.DataFrame(output_gate)
        df.to_csv(os.path.join(info_path, feature_name+'_output_gate.csv'), index=False, header=False)

        cell_gate=[]                    
        for index_seq in range( yee ):
            yee_hidden=np.matmul(w_hc.cpu().data.numpy(), hidden_state[:,index_seq]) + b_hc.cpu().data.numpy()
            x_temp=np.transpose( x[index_seq,:] )
            yee_input=np.matmul( w_ic.cpu().data.numpy(), x_temp) + b_ic.cpu().data.numpy()
            cell_gate+=[   torch.sigmoid(    torch.from_numpy(yee_hidden)+torch.from_numpy(yee_input)   ).numpy()  ]
        cell_gate=np.transpose(cell_gate)
        df=pd.DataFrame(cell_gate)
        df.to_csv(os.path.join(info_path, feature_name+'_cell_gate.csv'), index=False, header=False)

        return 

    def plot_heatmap(self, file_path, feature_name, plot_path):
        # file_path= os.path.join(file_path)
        file_list= os.listdir(file_path)
        plot_path=plot_path
        print(file_list)
        # print('len(file_list)= ', len(file_list))
        for i in range(len(file_list)):
            print(i, ' ')
            file_name=str(file_list[i])[:-4]
            print(file_name, ' ')
            df=pd.read_csv( file_path+'/'+ file_name+'.csv', sep=',',header=None )
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
            plt.savefig( plot_path+'/'+file_name+'.png' )

            plt.cla()
            plt.clf()
            plt.close()

        return

    def feature_visualization(self, file_path, plot_path, x, y):
        file_list= os.listdir(file_path)
        plot_path=plot_path
        x=x
        y=y

        for i in range(len(file_list)):
            file_name=str(file_list[i])[:-4]
            df=pd.read_csv( file_path+'/'+ file_name+'.csv', sep=',',header=None )
            rows=df.values

            sns.set(font_scale=1.5)
            plt.rcParams["figure.figsize"] = (16,9)

            # grid_kws = {"height_ratios": (.3, .02, .3, .3), "hspace": 0.01}

            f, (ax, ax3, ax4) = plt.subplots(3)
            # f.suptitle(file_name, fontsize=25)
            
            ax = sns.heatmap(rows, ax=ax,
                cbar=False,
                cmap="YlGnBu",
                yticklabels=False,
                # cbar_kws={"orientation": "horizontal"}
                )
            ax.set_title( file_name )
            
            ax3 = sns.heatmap(x.transpose(), ax=ax3, cbar=False, cmap='YlGnBu', yticklabels=False)
            ax3.set_title('Input Data')

            time=[]
            for i in range(y.shape[0]):
                time+=[i]

            ax4=plt.plot(time,y)
            # print('shape of y=', y.shape)
            plt.xlim([ time[0], time[-1] ])
            plt.ylabel('Velocity (mm/s)')
            plt.xlabel('Samples')

            plt.tight_layout()
            plt.savefig( plot_path+'/'+file_name+'.png' )
            plt.cla()
            plt.clf()
            plt.close()
        return