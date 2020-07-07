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
        self.GRU_1 = torch.nn.GRU( input_dim, hidden_dim, 1          , batch_first=True, bidirectional=True )
        self.GRU_2 = torch.nn.GRU( 2*hidden_dim, hidden_dim, layer_dim-1, batch_first=True, bidirectional=True )
        
        # Layer Normalization
        # self.input_LN_0 = torch.nn.LayerNorm( input_dim*max_timestep, elementwise_affine=True)
        self.input_LN_1 = torch.nn.LayerNorm( [max_timestep, 2*hidden_dim], elementwise_affine=True)
        self.input_LN_2 = torch.nn.LayerNorm( [max_timestep, 2*hidden_dim], elementwise_affine=True)

        # Readout layer
        # self.fc1 = torch.nn.Linear(hidden_dim, int(hidden_dim/2)) # one-directional
        # self.fc2 = torch.nn.Linear(int(hidden_dim/2), output_dim) # one-directional

        r = int( max_timestep/4 )
        da= int( hidden_dim/2 )

        self.W_s1 = torch.nn.Linear( 2*hidden_dim, da )
        self.W_s2 = torch.nn.Linear( da, r )

        self.fc_layer = torch.nn.Linear( r*2*hidden_dim, int(hidden_dim/2))
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

        # Layer Normalization 0
        # x=self.input_LN_0(x)

        x=x.view(x.size(0), -1, self.input_dim)

        out, _ = self.GRU_1(x)

        # Layer Normalization 1
        out = self.input_LN_1(out)

        out, _ = self.GRU_2(out)

        # Layer Normalization 2
        out = self.input_LN_2(out)

        attn_weight_matrix = self.attention_net(out)

        hidden_matrix = torch.bmm(attn_weight_matrix, out)

        out = torch.relu( self.fc_layer( hidden_matrix.view( -1 , hidden_matrix.size()[1]*hidden_matrix.size()[2] ) ) )

        out = self.label(out)


        return out, attn_weight_matrix
