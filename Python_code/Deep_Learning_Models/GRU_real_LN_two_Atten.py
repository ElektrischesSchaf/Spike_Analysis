# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

from .GRU_layernorm_cell import LayerNormGRUCell

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Real_Layer_GRU_bidir(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, max_timestep, layer_dim, output_dim):
        super(Real_Layer_GRU_bidir, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.input_dim=input_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.GRU_Cell_forward_1 = LayerNormGRUCell( input_dim      , hidden_dim    ,  bias=True )
        self.GRU_Cell_backward_1 = LayerNormGRUCell( input_dim      , hidden_dim    ,  bias=True )
        self.GRU_Cell_forward_2 = LayerNormGRUCell( hidden_dim*2      , hidden_dim    ,  bias=True )
        self.GRU_Cell_backward_2 = LayerNormGRUCell( hidden_dim*2      , hidden_dim   ,  bias=True )
        
        # Layer Normalization
        self.outside_layer_norm = torch.nn.LayerNorm( [max_timestep, 2*hidden_dim], elementwise_affine=True)
        

        # Readout layer
        # self.fc1 = torch.nn.Linear(hidden_dim, int(hidden_dim/2)) # one-directional
        # self.fc2 = torch.nn.Linear(int(hidden_dim/2), output_dim) # one-directional

        da= int( hidden_dim/2 )
        r = int( max_timestep/4 )

        da_hidden_units = 5
        r_hidden_units = 2

        self.W_s1_1 = torch.nn.Linear( 2*hidden_dim, da )
        self.W_s2_1 = torch.nn.Linear( da, r )

        self.W_s1_2 = torch.nn.Linear( max_timestep, da_hidden_units )
        self.W_s2_2 = torch.nn.Linear( da_hidden_units, r_hidden_units )

        self.fc_layer_1 = torch.nn.Linear( 2*r*hidden_dim, int(r*hidden_dim/2))
        self.fc_layer_2 = torch.nn.Linear( r_hidden_units*max_timestep, int(r_hidden_units*max_timestep/2))

        self.label = torch.nn.Linear( int(r*hidden_dim/2) + int(r_hidden_units*max_timestep/2), output_dim )


    def attention_net_temporal(self, gru_output):
        attn_weight_matrix = self.W_s2_1(torch.tanh(self.W_s1_1(gru_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        attn_weight_matrix = torch.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix

    def attention_net_hidden_units(self, gru_output):
        attn_weight_matrix = self.W_s2_2(torch.tanh(self.W_s1_2(gru_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        attn_weight_matrix = torch.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix

    def forward(self, x):
        x=x.view(x.size(0), -1, self.input_dim)

        # first layer
        # go forward
        h0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        h0=h0.to(device)


        hidden_state_list=[]
        out_list=[]
        for i , input_t in enumerate( x.chunk( x.size(1), dim=1 )):
            input_t=input_t.squeeze(1)
            # print('shape of input_t= ', input_t.size(), '\n')
            h0 = self.GRU_Cell_forward_1(input_t, h0)
            hidden_state_list+=[h0]

        hidden_state_list_1_forward = torch.stack(hidden_state_list, 0)
        hidden_state_list_1_forward = hidden_state_list_1_forward.permute(1,0,2)

        # go backward
        h0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        h0=h0.to(device)
        
        hidden_state_list=[]
        out_list=[]
        for i , input_t in reversed( list( enumerate( x.chunk( x.size(1), dim=1 ))) ):
            input_t=input_t.squeeze(1)
            h0 = self.GRU_Cell_backward_1(input_t, h0)
            hidden_state_list+=[h0]

        hidden_state_list_1_backward = torch.stack(hidden_state_list, 0)
        hidden_state_list_1_backward = hidden_state_list_1_backward.permute(1,0,2)

        hidden_state_list_1_backward = hidden_state_list_1_backward.flip(1)

        hidden_state_list_1_result = torch.cat( (hidden_state_list_1_forward, hidden_state_list_1_backward), 2 )

        # second layer
        # go forward
        h0 = torch.zeros( hidden_state_list_1_result.size(0), self.hidden_dim).requires_grad_() # one-directional
        h0=h0.to(device)

        hidden_state_list=[]
        out_list=[]

        for i , input_t in enumerate( hidden_state_list_1_result.chunk( hidden_state_list_1_result.size(1), dim=1 )):
            input_t=input_t.squeeze(1)
            h0 = self.GRU_Cell_forward_2(input_t, h0)
            hidden_state_list+=[h0]

        hidden_state_list = torch.stack(hidden_state_list, 0)
        hidden_state_list_2_forward = hidden_state_list.permute(1,0,2)

        # go backward
        h0 = torch.zeros( hidden_state_list_1_result.size(0), self.hidden_dim).requires_grad_() # one-directional
        h0=h0.to(device)
        
        hidden_state_list=[]
        out_list=[]
        # time steps
        for i , input_t in reversed( list( enumerate( hidden_state_list_1_result.chunk( x.size(1), dim=1 ))) ): #TODO
            input_t=input_t.squeeze(1)
            # print('shape of input_t= ', input_t.size(), '\n')
            h0 = self.GRU_Cell_backward_2(input_t, h0)
            hidden_state_list+=[h0]

        hidden_state_list_2_backward = torch.stack(hidden_state_list, 0)
        hidden_state_list_2_backward = hidden_state_list_2_backward.permute(1,0,2)

        hidden_state_list_2_backward = hidden_state_list_2_backward.flip(1) #TODO

        hidden_state_list_2_result = torch.cat((hidden_state_list_2_forward, hidden_state_list_2_backward), 2)

        hidden_state_list_2_result = self.outside_layer_norm(hidden_state_list_2_result)

        attn_weight_matrix_temporal = self.attention_net_temporal( hidden_state_list_2_result )

        hidden_state_list_2_result_transpose = hidden_state_list_2_result.clone().permute(0,2,1)
        attn_weight_matrix_hidden_units = self.attention_net_hidden_units( hidden_state_list_2_result_transpose )

        hidden_matrix_temporal = torch.bmm( attn_weight_matrix_temporal, hidden_state_list_2_result )
        hidden_matrix_units = torch.bmm( attn_weight_matrix_hidden_units, hidden_state_list_2_result_transpose )

        out_forward = torch.relu( self.fc_layer_1( hidden_matrix_temporal.view( -1 , hidden_matrix_temporal.size()[1]*hidden_matrix_temporal.size()[2] ) ) )
        out_hidden_units = torch.relu( self.fc_layer_2( hidden_matrix_units.view( -1 , hidden_matrix_units.size()[1]*hidden_matrix_units.size()[2] ) ) )


        out = self.label(  torch.cat( (out_forward, out_hidden_units), 1) )

        return out, attn_weight_matrix_temporal, attn_weight_matrix_hidden_units


class  Real_Layer_GRU_one_way(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, max_timestep, layer_dim, output_dim):
        super(Real_Layer_GRU_one_way, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.input_dim=input_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.GRU_Cell_forward_1 = LayerNormGRUCell( input_dim      , hidden_dim    ,  bias=True )
        self.GRU_Cell_forward_2 = LayerNormGRUCell( hidden_dim      , hidden_dim    ,  bias=True )

        # Layer Normalization
        self.input_LN_forward = torch.nn.LayerNorm( [max_timestep, hidden_dim], elementwise_affine=True)

        da= int( hidden_dim/2 )
        r = int( max_timestep/4 )

        da_hidden_units = 5
        r_hidden_units = 2

        self.W_s1_1 = torch.nn.Linear( hidden_dim, da )
        self.W_s2_1 = torch.nn.Linear( da, r )

        self.W_s1_2 = torch.nn.Linear( max_timestep, da_hidden_units )
        self.W_s2_2 = torch.nn.Linear( da_hidden_units, r_hidden_units )

        self.fc_layer_1 = torch.nn.Linear( r*hidden_dim, int(r*hidden_dim/2))
        self.fc_layer_2 = torch.nn.Linear( r_hidden_units*max_timestep, int(r_hidden_units*max_timestep/2))

        self.label = torch.nn.Linear( int(r*hidden_dim/2) + int(r_hidden_units*max_timestep/2), output_dim )

    def attention_net_temporal(self, gru_output):
        attn_weight_matrix = self.W_s2_1(torch.tanh(self.W_s1_1(gru_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        attn_weight_matrix = torch.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix

    def attention_net_hidden_units(self, gru_output):
        attn_weight_matrix = self.W_s2_2(torch.tanh(self.W_s1_2(gru_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        attn_weight_matrix = torch.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix
    
    def forward(self, x):

        x=x.view(x.size(0), -1, self.input_dim)

        # Initialize hidden state with zeros
        h0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        h0=h0.to(device)

        hidden_state_list=[]
        out_list=[]
        # time steps
        for i , input_t in enumerate( x.chunk( x.size(1), dim=1 )):
            input_t = input_t.squeeze(1)
            # print('shape of input_t= ', input_t.size(), '\n')
            h0 = self.GRU_Cell_forward_1(input_t, h0)
            hidden_state_list += [h0]

        hidden_state_list_1 = torch.stack(hidden_state_list, 0)

        hidden_state_list_1 = hidden_state_list_1.permute(1,0,2)

        # Initialize hidden state with zeros
        h0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        h0=h0.to(device)


        hidden_state_list=[]
        out_list=[]
        # time steps
        for i , input_t in enumerate( hidden_state_list_1.chunk( hidden_state_list_1.size(1), dim=1 )):
            input_t=input_t.squeeze(1)
            # print('shape of input_t= ', input_t.size(), '\n')
            h0 = self.GRU_Cell_forward_2(input_t, h0)
            hidden_state_list+=[h0]

        hidden_state_list = torch.stack(hidden_state_list, 0)

        hidden_state_list = hidden_state_list.permute(1,0,2)

        hidden_state_list = self.input_LN_forward(hidden_state_list)

        attn_weight_matrix_temporal = self.attention_net_temporal( hidden_state_list )

        hidden_state_list_transpose = hidden_state_list.clone().permute(0,2,1)
        attn_weight_matrix_hidden_units = self.attention_net_hidden_units( hidden_state_list_transpose )

        hidden_matrix_temporal = torch.bmm( attn_weight_matrix_temporal, hidden_state_list )
        hidden_matrix_units = torch.bmm( attn_weight_matrix_hidden_units, hidden_state_list_transpose )

        out_forward = torch.relu( self.fc_layer_1( hidden_matrix_temporal.view( -1 , hidden_matrix_temporal.size()[1]*hidden_matrix_temporal.size()[2] ) ) )
        out_hidden_units = torch.relu( self.fc_layer_2( hidden_matrix_units.view( -1 , hidden_matrix_units.size()[1]*hidden_matrix_units.size()[2] ) ) )


        out = self.label(  torch.cat( (out_forward, out_hidden_units), 1) )

        return out, attn_weight_matrix_temporal, attn_weight_matrix_hidden_units
