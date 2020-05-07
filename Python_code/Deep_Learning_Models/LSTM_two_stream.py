# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm_spike = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
        self.lstm_phase = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
        
        # Readout layer
        self.fc1 = torch.nn.Linear(hidden_dim, int(hidden_dim/2) ) # one-directional
        self.fc2 = torch.nn.Linear(int(hidden_dim/2), output_dim) # one-directional

        # self.fc1 = torch.nn.Linear(hidden_dim*2, hidden_dim) # bidirectional
        # self.fc2 = torch.nn.Linear(hidden_dim, output_dim) # bidirectional
    def forward(self, x):

        # x torch.Size([64, 96])

        x=x.unsqueeze(0)       

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_() # one-directional
        # h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_() # bidirectional
        h0=h0.to(device)
        h1=h0.clone()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_() # one-directional
        # c0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_() # bidirectional
        c0=c0.to(device)
        c1=c0.clone()
        # print('input dim= ', x.size(), '\n') # input dim=  torch.Size([1, 64, 96]) => batch_first=True, (batch_dim, seq_dim, feature_dim)

        # time steps
        out_spike, (hn, cn) = self.lstm_spike(x[:,:,:96], (h0,c0))
        out_phase, (hn, cn) = self.lstm_phase(x[:,:,96:], (h1,c1))

        '''
        Index hidden state of last time step
        out.size() --> 100, 28, 100
        out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        out.size() --> 100, 10
        '''

        out_spike=torch.relu( self.fc1(out_spike) )
        out_phase=torch.relu( self.fc1(out_phase) )

        out =torch.relu( self.fc1(torch.cat((out_spike,out_phase), 2) ) )
        out = self.fc2(out)
        out=out.squeeze(0)

        return out