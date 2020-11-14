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

# https://www.zhihu.com/question/385386895/answer/1133300166

class compute():
    def comp_and_save(self, input, hidden_state, feature_name, seq_length, save_path, w_ir, w_hr, w_iz, w_hz, w_in, w_hn, b_ir, b_hr, b_iz, b_hz, b_in, b_hn):
        yee = seq_length
        info_path = save_path
        feature_name = feature_name

        x=input[-1,:,:] # use the last one from the batch
        print('shape x=', x.shape, '\n')


        hidden_state = hidden_state.cpu().data.numpy()
        hidden_state = hidden_state[-1,:,:] # use the last one from the batch

        # hidden_state = np.transpose(hidden_state)
        print('hidden_state size 2: ', hidden_state.shape, '\n')

        df = pd.DataFrame(np.transpose(hidden_state))
        df.to_csv(os.path.join(info_path, feature_name+'_hidden_state.csv'), index=False, header=False)

        r_gate = []                    
        for index_seq in range( yee ):
            yee_hidden = np.matmul( w_hr.cpu().data.numpy(), hidden_state[ index_seq,: ] ) + b_hr.cpu().data.numpy()
            x_temp = np.transpose( x[index_seq,:] )
            yee_input = np.matmul( w_ir.cpu().data.numpy(), x_temp) + b_ir.cpu().data.numpy()
            r_gate += [   torch.sigmoid(    torch.from_numpy(yee_hidden)+torch.from_numpy(yee_input)   ).numpy()  ]
        r_gate = np.transpose(r_gate)
        df=pd.DataFrame(r_gate)
        df.to_csv(os.path.join(info_path, feature_name+'_r_gate.csv'), index=False, header=False)

        z_gate = []
        for index_seq in range( yee ):
            yee_hidden = np.matmul( w_hz.cpu().data.numpy(), hidden_state[ index_seq,: ] ) + b_hz.cpu().data.numpy()
            x_temp = np.transpose( x[index_seq,:] )
            yee_input = np.matmul( w_iz.cpu().data.numpy(), x_temp) + b_iz.cpu().data.numpy()
            z_gate += [   torch.sigmoid(    torch.from_numpy(yee_hidden)+torch.from_numpy(yee_input)   ).numpy()  ]
        z_gate = np.transpose(z_gate)
        df = pd.DataFrame(z_gate)
        df.to_csv(os.path.join(info_path, feature_name+'_z_gate.csv'), index=False, header=False)

        return 

    def plot_heatmap(self, file_path, plot_path):
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

    def feature_visualization(self, file_path, feature_name, plot_path, x, y, kinematic_type):
        file_list= os.listdir(file_path)
        plot_path=plot_path
        # x (batch, seq length, input dim)
        x=x[-1,:,:] # use the last one from the batch
        # y (batch, 2)
        if kinematic_type== 'x_and_y_pos':
            y=y[-20:,:] # use the last 20 from the batch
        else:
            y=y[-10:,:] # use the last 10 from the batch

        print('x.shape= ' ,x.shape, '\n') 
        print('y.shape= ' ,y.shape, '\n')

        for i in range(len(file_list)): # travel r_gate, z_gate, and hidden_states
            file_name=str(file_list[i])[:-4] 

            if file_name.startswith(feature_name):

                df=pd.read_csv( file_path+'/'+ file_name+'.csv', sep=',',header=None )
                rows=df.values

                sns.set(font_scale=1.5)
                plt.rcParams["figure.figsize"] = (16,9)

                # grid_kws = {"height_ratios": (.3, .02, .3, .3), "hspace": 0.01}

                f, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1,1,1],  "hspace":0.2 ,"left":0.1, "right":0.95, "top":0.95, "bottom":0.1} , constrained_layout=False)
                # f.suptitle(file_name, fontsize=25)
                
                time=[]
                for i in range(y.shape[0]):
                    time+=[i]

                ###############################################
                ax[0].plot(time,y)
                ax[0].title('Kinematics')
                # print('shape of y=', y.shape)
                ax[0].set_xlim([ time[0], time[-1] ])
                if kinematic_type=='x_and_y_pos':
                    ax[0].set_ylabel('Position (mm)')
                if kinematic_type=='x_and_y_vel':
                    ax[0].set_ylabel('Velocity (mm/s)')
                if kinematic_type=='x_and_y_acc':
                    ax[0].set_ylabel('Acceleration (mm/$s^2$)')
                ax[0].set_xlabel('')
                ax[0].set_xticks([])       

                ###############################################
                ax[1] = sns.heatmap(rows, ax=ax[1], cbar=False,  cmap="YlGnBu",  yticklabels=False, xticklabels=False )  # cbar_kws={"orientation": "horizontal"}
                ax[1].set_title( file_name )
                ax[1].set_ylabel('hidden units')
                ###############################################

                ax[2] = sns.heatmap(x.transpose(), ax=ax[2], cbar=False, cmap='YlGnBu', yticklabels=False, xticklabels=False)
                ax[2].set_title('Input Data')
                ax[2].set_ylabel('Features')
                ###############################################

                plt.savefig( plot_path+'/'+file_name+'.png' )
                plt.cla()
                plt.clf()
                plt.close()
        return