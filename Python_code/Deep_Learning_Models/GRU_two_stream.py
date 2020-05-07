# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class  GRUModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.GRU_spike = torch.nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
        self.GRU_phase = torch.nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
    
        # Readout layer
        self.fc1 = torch.nn.Linear(hidden_dim, int(hidden_dim/2) ) # one-directional
        self.fc2 = torch.nn.Linear(int(hidden_dim/2), output_dim) # one-directional

        # self.fc1 = torch.nn.Linear(hidden_dim*2, hidden_dim) # bidirectional
        # self.fc2 = torch.nn.Linear(hidden_dim, output_dim) # bidirectional
    
    def forward(self, x):

        # x torch.Size([64, 96])

        x=x.unsqueeze(0)       
        # m=torch.nn.LayerNorm( x.size()[:], elementwise_affine=False )
        # x=m(x)

        # print('input dim= ', x.size(), '\n') # input dim=  torch.Size([1, 64, 96]) => batch_first=True, (batch_dim, seq_dim, feature_dim)

        # time steps
        out_spike, _ = self.GRU_spike(x[:,:,:96])
        out_phase, _ = self.GRU_phase(x[:,:,96:])

        out_spike=torch.relu( self.fc1(out_spike) )
        out_phase=torch.relu( self.fc1(out_phase) )

        out =torch.relu( self.fc1(torch.cat((out_spike,out_phase), 2) ) )
        out = self.fc2(out)
        out=out.squeeze(0)

        return out
