# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

from .layer_norm_LSTM import LayerNormLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class  Real_Layer_LSTM(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, max_timestep, layer_dim, output_dim):
        super(Real_Layer_LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.input_dim=input_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.LSTM_1 = LayerNormLSTM( input_dim      , hidden_dim    , layer_dim         , bias=True, bidirectional=True )
        # self.LSTM_2 = LayerNormLSTM( 2*hidden_dim   , hidden_dim    , layer_dim-1   , bidirectional=True )
        
        # Layer Normalization
        # self.input_LN_0 = torch.nn.LayerNorm( input_dim*max_timestep, elementwise_affine=True)
        # self.input_LN_1 = torch.nn.LayerNorm( [max_timestep, 2*hidden_dim], elementwise_affine=True)
        # self.input_LN_2 = torch.nn.LayerNorm( [max_timestep, 2*hidden_dim], elementwise_affine=True)

        # Readout layer
        # self.fc1 = torch.nn.Linear(hidden_dim, int(hidden_dim/2)) # one-directional
        # self.fc2 = torch.nn.Linear(int(hidden_dim/2), output_dim) # one-directional

        r = int( max_timestep/4 )
        da= int( hidden_dim/2 )

        self.W_s1 = torch.nn.Linear( hidden_dim, da )
        self.W_s2 = torch.nn.Linear( da, r )

        self.fc_layer = torch.nn.Linear( r*hidden_dim, int(hidden_dim/2))
        self.label = torch.nn.Linear( int(hidden_dim), output_dim )

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

        x=x.permute(1,0,2)

        out, (hy, cy) = self.LSTM_1(x)

        print('out size= ', out.size(), '  hy size= ', hy.size() )
        breakpoint()
        out=out.permute(1,0,2)
        # print('shape of out= ', out.size())

        # Layer Normalization 1
        # out = self.input_LN_1(out)

        # y, out ,cy = self.LSTM_2(hy) #TODO

        # Layer Normalization 2
        # out = self.input_LN_2(out)

        attn_weight_matrix_forward = self.attention_net( out[:,:,:self.hidden_dim] )
        hidden_matrix_forward = torch.bmm( attn_weight_matrix_forward, out[:,:,:self.hidden_dim] )

        attn_weight_matrix_backward = self.attention_net( out[:,:,-self.hidden_dim:] )
        hidden_matrix_backward = torch.bmm( attn_weight_matrix_backward, out[:,:,-self.hidden_dim:] )

        out_forward = torch.relu( self.fc_layer( hidden_matrix_forward.view( -1 , hidden_matrix_forward.size()[1]*hidden_matrix_forward.size()[2] ) ) )
        out_backward = torch.relu( self.fc_layer( hidden_matrix_backward.view( -1 , hidden_matrix_backward.size()[1]*hidden_matrix_backward.size()[2] ) ) )

        out = self.label( torch.cat( (out_forward, out_backward), 1) )
        # attn_weight_matrix = attn_weight_matrix_forward + attn_weight_matrix_backward

        return out, attn_weight_matrix_forward, attn_weight_matrix_backward