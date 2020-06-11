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
        self.input_dim=input_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.GRU = torch.nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
        
        # Readout layer
        self.fc1 = torch.nn.Linear(hidden_dim, int(hidden_dim/2)) # one-directional
        self.fc2 = torch.nn.Linear(int(hidden_dim/2), output_dim) # one-directional

        r = 1
        da= 50

        self.W_s1 = torch.nn.Linear(hidden_dim, da)
        self.W_s2 = torch.nn.Linear(da, r)

        self.fc_layer = torch.nn.Linear( r*hidden_dim, int(hidden_dim/2))
        self.label = torch.nn.Linear( int(hidden_dim/2), output_dim )

    def attention_net(self, gru_output):
        # lstm_output=lstm_output.permute(0, 2, 1)
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(gru_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        attn_weight_matrix = torch.softmax(attn_weight_matrix, dim=2)
        # print('shape of attn_weight_matrix= ', attn_weight_matrix.size())
        # print(attn_weight_matrix)
        return attn_weight_matrix
    
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


        attn_weight_matrix = self.attention_net(out)

        hidden_matrix = torch.bmm(attn_weight_matrix, out)
        # print('shape of hidden_matrix= ', hidden_matrix.size())
        out = (self.fc_layer( hidden_matrix.view( -1 , hidden_matrix.size()[1]*hidden_matrix.size()[2] ) ) )

        out = self.label(out)


        return out
