import numpy as np
import pandas as pd
import os
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
        df.to_csv(os.path.join(info_path, feature_name+'_hidden.csv'), index=False, header=False)

        if cell_state:
            cell_state=cell_state.cpu().data.numpy()
            cell_state=np.transpose(cell_state)
            df=pd.DataFrame(cell_state)
            df.to_csv(os.path.join(info_path, feature_name+'cell_state.csv'), index=False, header=False)

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
