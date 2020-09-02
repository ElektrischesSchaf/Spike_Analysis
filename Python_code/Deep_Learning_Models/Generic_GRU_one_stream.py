# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class  GRUModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, max_timestep, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.input_dim=input_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.GRU = torch.nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
        
        # Readout layer
        self.fc1 = torch.nn.Linear(hidden_dim, int(hidden_dim/2)) # one-directional
        self.fc2 = torch.nn.Linear(int(hidden_dim/2), output_dim) # one-directional

        # self.fc1 = torch.nn.Linear(hidden_dim*2, hidden_dim) # bidirectional
        # self.fc2 = torch.nn.Linear(hidden_dim, output_dim) # bidirectional
    
    def forward(self, x):

        # x torch.Size([batch size, feature num * orders])
        # print('x size 1= ', x.size())
        x=x.view(x.size(0), -1, self.input_dim)
        # print('x size 2= ', x.size())
        # m=torch.nn.LayerNorm( x.size()[:], elementwise_affine=False )
        # x=m(x)

        # print('input dim= ', x.size(), '\n') # input dim=  torch.Size([1, 64, 96]) => batch_first=True, (batch_dim, seq_dim, feature_dim)
        # print('yee shape of x= ', x.size())
        # time steps
        # print('real input shape= ', x.size(), '\n')
        out, _ = self.GRU(x)
        # print('out size 1= ', out.size())
        out = torch.relu(self.fc1(out[:,-1,:] ))
        # print('out size 2= ', out.size())
        out = self.fc2(out)
        # print('out size 3= ', out.size())


        return out
