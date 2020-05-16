# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class  LSTMCELLModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMCELLModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstmcell = torch.nn.LSTMCell(input_dim, hidden_dim)
        
        # Readout layer
        self.fc1 = torch.nn.Linear(hidden_dim, int(hidden_dim/2)) # one-directional
        self.fc2 = torch.nn.Linear(int(hidden_dim/2), output_dim) # one-directional

        # self.fc1 = torch.nn.Linear(hidden_dim*2, hidden_dim) # bidirectional
        # self.fc2 = torch.nn.Linear(hidden_dim, output_dim) # bidirectional
    
    def forward(self, x):

        # x torch.Size([64, 96])
        x=x.unsqueeze(0)   
        # x torch.Size([1, 64, 96])


        # Initialize hidden state with zeros
        h0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        # h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_() # bidirectional
        h0=h0.to(device)

        # Initialize cell state
        c0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        # c0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_() # bidirectional
        c0=c0.to(device)

        out_list=[]
        # time steps
        for i , input_t in enumerate( x.chunk( x.size(1), dim=1 )):
            input_t=input_t.squeeze(1)
            # print('shape of input_t= ', input_t.size(), '\n')
            (h0, c0) = self.lstmcell(input_t, (h0, c0))
            out=torch.relu(self.fc1(h0))
            out=self.fc2(out)
            # print('out size= ', out.size(), '\n')
            out_list+=[out]

        out_list = torch.stack(out_list, 0)
        # print('out_list size= ', out_list.size(), '\n')

        out=out_list.squeeze(1)

        return out
