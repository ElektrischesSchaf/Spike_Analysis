# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class  GRUModel(torch.nn.Module):

    def __init__(self, firing_rate_input_dim, result_conv_input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()

        self.firing_rate_input_dim=firing_rate_input_dim
        self.result_conv_input_dim=result_conv_input_dim

        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.GRU_spike = torch.nn.GRU(firing_rate_input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
        self.GRU_conv_average_phase = torch.nn.GRU(result_conv_input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
        self.GRU_conv_average_sync = torch.nn.GRU(result_conv_input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)

        # Readout layer
        self.fc1 = torch.nn.Linear(hidden_dim, int(hidden_dim/2) ) # one-directional
        self.fc2 = torch.nn.Linear(int(hidden_dim/2), int(hidden_dim/4) ) # one-directional

        # After concatenation
        self.fc3 = torch.nn.Linear(int(int(hidden_dim/4)*3), int((int(hidden_dim/4)*3)/2)  ) # one-directional
        self.fc4 = torch.nn.Linear(int((int(hidden_dim/4)*3)/2), output_dim) # one-directional
        # self.fc1 = torch.nn.Linear(hidden_dim*2, hidden_dim) # bidirectional
        # self.fc2 = torch.nn.Linear(hidden_dim, output_dim) # bidirectional
    
    def forward(self, x):

        # x torch.Size([64, 96+64+64])

        x=x.unsqueeze(0)       
        # m=torch.nn.LayerNorm( x.size()[:], elementwise_affine=False )
        # x=m(x)

        # print('input dim= ', x.size(), '\n') # input dim=  torch.Size([1, 64, 96]) => batch_first=True, (batch_dim, seq_dim, feature_dim)

        # time steps
        out_spike, _ = self.GRU_spike(x[ : , : , : self.firing_rate_input_dim ])
        out_conv_average_phase, _ = self.GRU_conv_average_phase(x[ : , : , self.firing_rate_input_dim : self.firing_rate_input_dim+self.result_conv_input_dim ])
        out_conv_average_sync, _ = self.GRU_conv_average_sync(x[ : , : , self.firing_rate_input_dim+self.result_conv_input_dim : self.firing_rate_input_dim+self.result_conv_input_dim+self.result_conv_input_dim ])

        out_spike=torch.relu( self.fc1(out_spike) )
        out_conv_average_phase=torch.relu( self.fc1(out_conv_average_phase) )
        out_conv_average_sync=torch.relu( self.fc1(out_conv_average_sync) )

        out_spike=torch.relu( self.fc2(out_spike) )
        out_conv_average_phase=torch.relu( self.fc2(out_conv_average_phase) )
        out_conv_average_sync=torch.relu( self.fc2(out_conv_average_sync) )

        out =torch.relu( self.fc3(torch.cat((out_spike, out_conv_average_phase, out_conv_average_sync), 2) ) )
        out = self.fc4(out)
        out=out.squeeze(0)

        return out
