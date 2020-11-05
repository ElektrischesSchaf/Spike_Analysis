# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class  GRUModel_bidir(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, max_timestep, layer_dim, output_dim):
        super(GRUModel_bidir, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.input_dim=input_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        r = int( max_timestep/2 )
        da= int( hidden_dim/2 )

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.GRU_1 = torch.nn.GRU( input_dim, hidden_dim,  2        , batch_first=True, bidirectional=True )

        # Layer Normalization
        # self.input_LN_0 = torch.nn.LayerNorm( input_dim*max_timestep, elementwise_affine=True)
        # self.input_LN_1 = torch.nn.LayerNorm( [max_timestep, hidden_dim], elementwise_affine=True)
        # self.input_LN_2 = torch.nn.LayerNorm( [max_timestep, hidden_dim], elementwise_affine=True)
        self.LN_in_atten = torch.nn.LayerNorm([max_timestep, da], elementwise_affine=False)


        self.W_s1 = torch.nn.Linear( 2*hidden_dim, da )
        self.W_s2 = torch.nn.Linear( da, r )

        self.fc_layer_x = torch.nn.Linear( 2*r*hidden_dim, int(hidden_dim))
        self.label_x = torch.nn.Linear( int(hidden_dim), 1 )

        self.fc_layer_y = torch.nn.Linear( 2*r*hidden_dim, int(hidden_dim))
        self.label_y = torch.nn.Linear( int(hidden_dim), 1 )

    def attention_net(self, gru_output):
        # lstm_output=lstm_output.permute(0, 2, 1)
        attn_weight_matrix = self.W_s2(torch.tanh( self.LN_in_atten(self.W_s1(gru_output))))
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


        attn_weight_matrix = self.attention_net(out)

        hidden_matrix = torch.bmm(attn_weight_matrix, out)

        hidden_matrix_x = hidden_matrix.clone()
        hidden_matrix_y = hidden_matrix.clone()

        out_x = torch.relu( self.fc_layer_x( hidden_matrix_x.view( -1 , hidden_matrix_x.size()[1]*hidden_matrix_x.size()[2] ) ) )
        out_y = torch.relu( self.fc_layer_y( hidden_matrix_y.view( -1 , hidden_matrix_y.size()[1]*hidden_matrix_y.size()[2] ) ) )

        out_x = self.label_x(out_x)
        out_y = self.label_y(out_y)

        out = torch.cat( ( out_x, out_y), 1)

        return out, attn_weight_matrix

class  GRUModel_oneway(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, max_timestep, layer_dim, output_dim):
        super(GRUModel_oneway, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.input_dim=input_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        r = int( max_timestep/2 )
        da= int( hidden_dim/2 )

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.GRU_1 = torch.nn.GRU( input_dim, hidden_dim, 2          , batch_first=True, bidirectional=False )
        
        # Layer Normalization
        # self.input_LN_0 = torch.nn.LayerNorm( input_dim*max_timestep, elementwise_affine=True)
        # self.input_LN_1 = torch.nn.LayerNorm( [max_timestep, hidden_dim], elementwise_affine=True)
        # self.input_LN_2 = torch.nn.LayerNorm( [max_timestep, hidden_dim], elementwise_affine=True)
        self.LN_in_atten = torch.nn.LayerNorm([max_timestep, da], elementwise_affine=False)


        self.W_s1 = torch.nn.Linear( hidden_dim, da )
        self.W_s2 = torch.nn.Linear( da, r )

        self.fc_layer_x = torch.nn.Linear( r*hidden_dim, int(hidden_dim/2))
        self.label_x = torch.nn.Linear( int(hidden_dim/2), 1 )

        self.fc_layer_y = torch.nn.Linear( r*hidden_dim, int(hidden_dim/2))
        self.label_y = torch.nn.Linear( int(hidden_dim/2), 1 )

    def attention_net(self, gru_output):
        # lstm_output=lstm_output.permute(0, 2, 1)
        attn_weight_matrix = self.W_s2(torch.tanh( self.LN_in_atten(self.W_s1(gru_output))))
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


        attn_weight_matrix = self.attention_net(out)

        hidden_matrix = torch.bmm(attn_weight_matrix, out)

        hidden_matrix_x = hidden_matrix.clone()
        hidden_matrix_y = hidden_matrix.clone()

        out_x = torch.relu( self.fc_layer_x( hidden_matrix_x.view( -1 , hidden_matrix_x.size()[1]*hidden_matrix_x.size()[2] ) ) )
        out_y = torch.relu( self.fc_layer_y( hidden_matrix_y.view( -1 , hidden_matrix_y.size()[1]*hidden_matrix_y.size()[2] ) ) )

        out_x = self.label_x(out_x)
        out_y = self.label_y(out_y)

        out = torch.cat( ( out_x, out_y), 1)

        return out, attn_weight_matrix


