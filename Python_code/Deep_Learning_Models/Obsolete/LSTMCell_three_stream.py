# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class  LSTMCell(torch.nn.Module):

    def __init__(self, firing_rate_input_dim, result_conv_input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMCell, self).__init__()

        self.firing_rate_input_dim=firing_rate_input_dim
        self.result_conv_input_dim=result_conv_input_dim

        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim


        self.lstm_spike_1 = torch.nn.LSTMCell(firing_rate_input_dim, hidden_dim)
        self.lstm_conv_average_phase_1 = torch.nn.LSTMCell(result_conv_input_dim, hidden_dim)
        self.lstm_conv_average_sync_1 = torch.nn.LSTMCell(result_conv_input_dim, hidden_dim)

        self.lstm_spike_2 = torch.nn.LSTMCell(hidden_dim, hidden_dim)
        self.lstm_conv_average_phase_2 = torch.nn.LSTMCell(hidden_dim, hidden_dim)
        self.lstm_conv_average_sync_2 = torch.nn.LSTMCell(hidden_dim, hidden_dim)

        # Readout layer
        self.fc1 = torch.nn.Linear(hidden_dim, int(hidden_dim/2) ) # one-directional
        self.fc2 = torch.nn.Linear(int(hidden_dim/2), int(hidden_dim/4) ) # one-directional

        # After concatenation
        self.fc3 = torch.nn.Linear(int(int(hidden_dim/4)*3), int((int(hidden_dim/4)*3)/2)  ) # one-directional
        self.fc4 = torch.nn.Linear(int((int(hidden_dim/4)*3)/2), output_dim) # one-directional

    
    def forward(self, x):

        # x torch.Size([64, 96+64+64])

        x=x.unsqueeze(0)       
        # m=torch.nn.LayerNorm( x.size()[:], elementwise_affine=False )
        # x=m(x)

        # print('input dim= ', x.size(), '\n') # input dim=  torch.Size([1, 64, 96]) => batch_first=True, (batch_dim, seq_dim, feature_dim)

        # time steps

        out_all=[]
        out_spike_hidden=[]
        out_conv_average_phase_hidden=[]
        out_conv_average_sync_hidden=[]
        out_spike_cell=[]
        out_conv_average_phase_cell=[]
        out_conv_average_sync_cell=[]

        h_spike_1 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_spike_1 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_().to(device)
        h_spike_2 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_spike_2 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_().to(device)

        h_conv_average_phase_1 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_conv_average_phase_1 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_().to(device)
        h_conv_average_phase_2 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_conv_average_phase_2 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_().to(device)

        h_conv_average_sync_1 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_conv_average_sync_1 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_().to(device)
        h_conv_average_sync_2 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_conv_average_sync_2 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_().to(device)

        for i , input_t in enumerate( x.chunk( x.size(1), dim=1 )):
            input_t=input_t.squeeze(1)
            # input_t size = (1, 224) if kernel size = 3

            # No. 1 layer
            h_spike_1, c_spike_1 = self.lstm_spike_1(input_t[ : , : self.firing_rate_input_dim ], (h_spike_1, c_spike_1))
            h_conv_average_phase_1, c_conv_average_phase_1 = self.lstm_conv_average_phase_1(input_t[ : , self.firing_rate_input_dim : self.firing_rate_input_dim+self.result_conv_input_dim ], (h_conv_average_phase_1, c_conv_average_phase_1))
            h_conv_average_sync_1, c_conv_average_sync_1 = self.lstm_conv_average_sync_1(input_t[ : , self.firing_rate_input_dim+self.result_conv_input_dim : self.firing_rate_input_dim+self.result_conv_input_dim+self.result_conv_input_dim ], (h_conv_average_sync_1, c_conv_average_sync_1))
            
            # No. 2 layer
            h_spike_2, c_spike_2 = self.lstm_spike_2(h_spike_1, (h_spike_2, c_spike_2))
            h_conv_average_phase_2, c_conv_average_phase_2 = self.lstm_conv_average_phase_2(h_conv_average_phase_1, (h_conv_average_phase_2, c_conv_average_phase_2))
            h_conv_average_sync_2, c_conv_average_sync_2 = self.lstm_conv_average_sync_2(h_conv_average_sync_1, (h_conv_average_sync_2, c_conv_average_sync_2))

            # FC 
            out_spike=torch.relu( self.fc1(h_spike_2) )
            out_conv_average_phase=torch.tanh( self.fc1(h_conv_average_phase_2) )
            out_conv_average_sync=torch.tanh( self.fc1(h_conv_average_sync_2) )

            # FC
            out_spike=torch.relu( self.fc2(out_spike) )
            out_conv_average_phase=torch.tanh( self.fc2(out_conv_average_phase) )
            out_conv_average_sync=torch.tanh( self.fc2(out_conv_average_sync) )

            # FC and final output
            out = ( self.fc3(torch.cat((out_spike, out_conv_average_phase, out_conv_average_sync), 1) ) )
            out = self.fc4(out)

            # Record the states
            out_all+=[out]
            out_spike_hidden+=[h_spike_1]
            out_conv_average_phase_hidden+=[h_conv_average_phase_1]
            out_conv_average_sync_hidden+=[h_conv_average_sync_1]

        out_all=torch.stack(out_all, 0)
        out_spike_hidden=torch.stack(out_spike_hidden, 0)
        out_conv_average_phase_hidden=torch.stack(out_conv_average_phase_hidden, 0)
        out_conv_average_sync_hidden=torch.stack(out_conv_average_sync_hidden, 0)

        out=out_all.squeeze(1)
        out_spike_hidden=out_spike_hidden.squeeze(1)
        out_conv_average_phase_hidden=out_conv_average_phase_hidden.squeeze(1)
        out_conv_average_sync_hidden=out_conv_average_sync_hidden.squeeze(1)
        # print('size= ', out.size(), ' ', out_spike_hidden.size(), ' ', out_conv_average_phase_hidden.size(), ' ', out_conv_average_sync_hidden.size())
        return out, out_spike_hidden, out_conv_average_phase_hidden, out_conv_average_sync_hidden
