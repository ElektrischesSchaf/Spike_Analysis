# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

from .GRU_layernorm_cell import LayerNormGRUCell

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class  Real_Layer_GRU_bidir_sep(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, max_timestep, layer_dim, output_dim):
        super(Real_Layer_GRU_bidir_sep, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.input_dim=input_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.GRU_Cell_forward_1 = LayerNormGRUCell( input_dim      , hidden_dim    ,  bias=True )
        self.GRU_Cell_backward_1 = LayerNormGRUCell( input_dim      , hidden_dim    ,  bias=True )
        self.GRU_Cell_forward_2 = LayerNormGRUCell( hidden_dim      , hidden_dim    ,  bias=True )
        self.GRU_Cell_backward_2 = LayerNormGRUCell( hidden_dim      , hidden_dim    ,  bias=True )
        
        # Layer Normalization
        # self.input_LN_0 = torch.nn.LayerNorm( input_dim*max_timestep, elementwise_affine=True)
        self.input_LN_forward = torch.nn.LayerNorm( [max_timestep, hidden_dim], elementwise_affine=True)
        self.input_LN_backward = torch.nn.LayerNorm( [max_timestep, hidden_dim], elementwise_affine=True)

        # Readout layer
        # self.fc1 = torch.nn.Linear(hidden_dim, int(hidden_dim/2)) # one-directional
        # self.fc2 = torch.nn.Linear(int(hidden_dim/2), output_dim) # one-directional

        r = int( max_timestep/4 )
        da= int( hidden_dim/2 )

        self.W_s1_1 = torch.nn.Linear( hidden_dim, da )
        self.W_s2_1 = torch.nn.Linear( da, r )
        self.fc_layer_1 = torch.nn.Linear( r*hidden_dim, int(hidden_dim/2))

        self.W_s1_2 = torch.nn.Linear( hidden_dim, da )
        self.W_s2_2 = torch.nn.Linear( da, r )
        self.fc_layer_2 = torch.nn.Linear( r*hidden_dim, int(hidden_dim/2))

        self.label = torch.nn.Linear( int(hidden_dim), output_dim )

    def attention_net_1(self, gru_output):
        # GRU_output=GRU_output.permute(0, 2, 1)
        attn_weight_matrix = self.W_s2_1(torch.tanh(self.W_s1_1(gru_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        attn_weight_matrix = torch.softmax(attn_weight_matrix, dim=2)
        # print('shape of attn_weight_matrix= ', attn_weight_matrix.size())
        # print(attn_weight_matrix)
        return attn_weight_matrix

    def attention_net_2(self, gru_output):
        # GRU_output=GRU_output.permute(0, 2, 1)
        attn_weight_matrix = self.W_s2_2(torch.tanh(self.W_s1_2(gru_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        attn_weight_matrix = torch.softmax(attn_weight_matrix, dim=2)
        # print('shape of attn_weight_matrix= ', attn_weight_matrix.size())
        # print(attn_weight_matrix)
        return attn_weight_matrix
    
    def forward(self, x):

        x=x.view(x.size(0), -1, self.input_dim)

        # Initialize hidden state with zeros
        h0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        h0=h0.to(device)

        # Initialize cell state
        c0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        c0=c0.to(device)

        hidden_state_list=[]
        cell_state_list=[]
        out_list=[]
        # time steps
        for i , input_t in enumerate( x.chunk( x.size(1), dim=1 )):
            input_t=input_t.squeeze(1)
            # print('shape of input_t= ', input_t.size(), '\n')
            h0, c0 = self.GRU_Cell_forward_1(input_t, h0, c0)
            hidden_state_list+=[h0]
            cell_state_list+=[c0]

        hidden_state_list_1 = torch.stack(hidden_state_list, 0)
        cell_state_list = torch.stack(cell_state_list, 0)

        hidden_state_list_1 = hidden_state_list_1.permute(1,0,2)
        cell_state_list = cell_state_list.permute(1,0,2)

        # Initialize hidden state with zeros
        h0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        h0=h0.to(device)

        # Initialize cell state
        c0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        c0=c0.to(device)

        hidden_state_list=[]
        cell_state_list=[]
        out_list=[]
        # time steps
        for i , input_t in enumerate( hidden_state_list_1.chunk( hidden_state_list_1.size(1), dim=1 )):
            input_t=input_t.squeeze(1)
            # print('shape of input_t= ', input_t.size(), '\n')
            h0, c0 = self.GRU_Cell_forward_2(input_t, h0, c0)
            hidden_state_list+=[h0]
            cell_state_list+=[c0]

        hidden_state_list = torch.stack(hidden_state_list, 0)
        cell_state_list = torch.stack(cell_state_list, 0)

        hidden_state_list = hidden_state_list.permute(1,0,2)
        cell_state_list = cell_state_list.permute(1,0,2)

        hidden_state_list = self.input_LN_forward(hidden_state_list)

        attn_weight_matrix_forward = self.attention_net_1( hidden_state_list )
        hidden_matrix_forward = torch.bmm( attn_weight_matrix_forward, hidden_state_list )


        # Initialize hidden state with zeros
        h0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        h0=h0.to(device)

        # Initialize cell state
        c0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        c0=c0.to(device)

        hidden_state_list=[]
        cell_state_list=[]
        out_list=[]
        # time steps
        for i , input_t in reversed( list( enumerate( x.chunk( x.size(1), dim=1 ))) ):
            input_t=input_t.squeeze(1)
            # print('shape of input_t= ', input_t.size(), '\n')
            h0, c0 = self.GRU_Cell_backward_1(input_t, h0, c0)
            hidden_state_list+=[h0]
            cell_state_list+=[c0]

        hidden_state_list_1 = torch.stack(hidden_state_list, 0)
        cell_state_list = torch.stack(cell_state_list, 0)

        hidden_state_list_1 = hidden_state_list_1.permute(1,0,2)
        cell_state_list = cell_state_list.permute(1,0,2)

        # Initialize hidden state with zeros
        h0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        h0=h0.to(device)

        # Initialize cell state
        c0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        c0=c0.to(device)

        hidden_state_list=[]
        cell_state_list=[]
        out_list=[]
        # time steps
        for i , input_t in enumerate( hidden_state_list_1.chunk( hidden_state_list_1.size(1), dim=1 )):
            input_t=input_t.squeeze(1)
            # print('shape of input_t= ', input_t.size(), '\n')
            h0, c0 = self.GRU_Cell_backward_2(input_t, h0, c0)
            hidden_state_list+=[h0]
            cell_state_list+=[c0]

        hidden_state_list = torch.stack(hidden_state_list, 0)
        cell_state_list = torch.stack(cell_state_list, 0)

        hidden_state_list = hidden_state_list.permute(1,0,2)
        cell_state_list = cell_state_list.permute(1,0,2)
    
        hidden_state_list = hidden_state_list.flip(1)

        hidden_state_list = self.input_LN_backward(hidden_state_list)

        attn_weight_matrix_backward = self.attention_net_2( hidden_state_list )
        hidden_matrix_backward = torch.bmm( attn_weight_matrix_backward, hidden_state_list )

        out_forward = torch.relu( self.fc_layer_1( hidden_matrix_forward.view( -1 , hidden_matrix_forward.size()[1]*hidden_matrix_forward.size()[2] ) ) )
        out_backward = torch.relu( self.fc_layer_2( hidden_matrix_backward.view( -1 , hidden_matrix_backward.size()[1]*hidden_matrix_backward.size()[2] ) ) )

        out = self.label( torch.cat( (out_forward, out_backward), 1) )
        # attn_weight_matrix = attn_weight_matrix_forward + attn_weight_matrix_backward

        return out, attn_weight_matrix_forward, attn_weight_matrix_backward


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

        r = int( max_timestep/4 )
        da= int( hidden_dim/2 )

        self.W_s1_1 = torch.nn.Linear( 2*hidden_dim, da )
        self.W_s2_1 = torch.nn.Linear( da, r )
        self.fc_layer_1 = torch.nn.Linear( 2*r*hidden_dim, int(hidden_dim))

        self.label = torch.nn.Linear( int(hidden_dim), output_dim )

    def attention_net(self, gru_output):
        # GRU_output=GRU_output.permute(0, 2, 1)
        attn_weight_matrix = self.W_s2_1(torch.tanh(self.W_s1_1(gru_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        attn_weight_matrix = torch.softmax(attn_weight_matrix, dim=2)
        # print('shape of attn_weight_matrix= ', attn_weight_matrix.size())
        # print(attn_weight_matrix)
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

        attn_weight_matrix = self.attention_net( hidden_state_list_2_result )
        hidden_matrix_ = torch.bmm( attn_weight_matrix, hidden_state_list_2_result )


        out = torch.relu( self.fc_layer_1( hidden_matrix_.view( -1 , hidden_matrix_.size()[1]*hidden_matrix_.size()[2] ) ) )

        out = self.label( out )

        return out, attn_weight_matrix

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

        r = int( max_timestep/4 )
        da= int( hidden_dim/2 )

        self.W_s1_1 = torch.nn.Linear( hidden_dim, da )
        self.W_s2_1 = torch.nn.Linear( da, r )
        self.LN_in_atten = torch.nn.LayerNorm([max_timestep, da], elementwise_affine=False)
        self.fc_layer_1 = torch.nn.Linear( r*hidden_dim, int(hidden_dim/2))

        self.label = torch.nn.Linear( int(hidden_dim/2), output_dim )

    def attention_net_1(self, gru_output):
        attn_weight_matrix = self.W_s2_1( torch.tanh( self.LN_in_atten( self.W_s1_1(gru_output) )  ))
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

        attn_weight_matrix_forward = self.attention_net_1( hidden_state_list )
        hidden_matrix_forward = torch.bmm( attn_weight_matrix_forward, hidden_state_list )

        out_forward = torch.relu( self.fc_layer_1( hidden_matrix_forward.view( -1 , hidden_matrix_forward.size()[1]*hidden_matrix_forward.size()[2] ) ) )

        out = self.label( out_forward )

        return out, attn_weight_matrix_forward

class Real_Layer_GRU_one_way_two_stream(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, max_timestep, layer_dim, output_dim, sorted_unit_numbers):
        super(Real_Layer_GRU_one_way_two_stream, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.input_dim=input_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        self.sorted_unit_numbers=sorted_unit_numbers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.GRU_Cell_forward_1 = LayerNormGRUCell( sorted_unit_numbers*96      , hidden_dim    ,  bias=True )
        self.GRU_Cell_forward_2 = LayerNormGRUCell( hidden_dim      , hidden_dim    ,  bias=True )

        self.GRU_Cell_forward_3 = LayerNormGRUCell( sorted_unit_numbers*96      , hidden_dim    ,  bias=True )
        self.GRU_Cell_forward_4 = LayerNormGRUCell( hidden_dim      , hidden_dim    ,  bias=True )
        # Layer Normalization
        self.input_LN_forward_1 = torch.nn.LayerNorm( [max_timestep, hidden_dim], elementwise_affine=True)
        self.input_LN_forward_2 = torch.nn.LayerNorm( [max_timestep, hidden_dim], elementwise_affine=True)

        r = int( max_timestep/4 )
        da= int( hidden_dim/2 )

        self.W_s1_1 = torch.nn.Linear( hidden_dim, da )
        self.W_s2_1 = torch.nn.Linear( da, r )

        self.W_s1_2 = torch.nn.Linear( hidden_dim, da )
        self.W_s2_2 = torch.nn.Linear( da, r )

        self.fc_layer_1 = torch.nn.Linear( r*hidden_dim, int(hidden_dim/2))
        self.fc_layer_2 = torch.nn.Linear( r*hidden_dim, int(hidden_dim/2))
        self.label = torch.nn.Linear( int(hidden_dim), output_dim )

    def attention_net_1(self, gru_output):
        attn_weight_matrix = self.W_s2_1(torch.tanh(self.W_s1_1(gru_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        attn_weight_matrix = torch.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix

    def attention_net_2(self, gru_output):
        attn_weight_matrix = self.W_s2_2(torch.tanh(self.W_s1_2(gru_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        attn_weight_matrix = torch.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix

    def forward(self, x):

        x = x.view(x.size(0), -1, self.input_dim)

        x_M1 = x[:,:,:self.sorted_unit_numbers*96]
        x_S1 = x[:,:,self.sorted_unit_numbers*96:]

        # layer one
        h0 = torch.zeros( x_M1.size(0), self.hidden_dim).requires_grad_() # one-directional
        h0=h0.to(device)
        hidden_state_list=[]
        for i , input_t in enumerate( x_M1.chunk( x_M1.size(1), dim=1 )):
            input_t = input_t.squeeze(1)
            h0 = self.GRU_Cell_forward_1(input_t, h0)
            hidden_state_list += [h0]
        hidden_state_list_1 = torch.stack(hidden_state_list, 0)
        hidden_state_list_1 = hidden_state_list_1.permute(1,0,2)

        # layer two
        h0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        h0=h0.to(device)
        hidden_state_list=[]
        for i , input_t in enumerate( hidden_state_list_1.chunk( hidden_state_list_1.size(1), dim=1 )):
            input_t=input_t.squeeze(1)
            h0 = self.GRU_Cell_forward_2(input_t, h0)
            hidden_state_list += [h0]
        hidden_state_list = torch.stack(hidden_state_list, 0)
        hidden_state_list = hidden_state_list.permute(1,0,2)

        hidden_state_list = self.input_LN_forward_1(hidden_state_list)
        attn_weight_matrix_forward_M1 = self.attention_net_1( hidden_state_list )
        hidden_matrix_forward_M1 = torch.bmm( attn_weight_matrix_forward_M1, hidden_state_list )
        out_forward_M1 = torch.relu( self.fc_layer_1( hidden_matrix_forward_M1.view( -1 , hidden_matrix_forward_M1.size()[1]*hidden_matrix_forward_M1.size()[2] ) ) )

        # layer one
        h0 = torch.zeros( x_S1.size(0), self.hidden_dim).requires_grad_() # one-directional
        h0=h0.to(device)
        hidden_state_list=[]
        for i , input_t in enumerate( x_S1.chunk( x_S1.size(1), dim=1 )):
            input_t = input_t.squeeze(1)
            h0 = self.GRU_Cell_forward_3(input_t, h0)
            hidden_state_list += [h0]
        hidden_state_list_1 = torch.stack(hidden_state_list, 0)
        hidden_state_list_1 = hidden_state_list_1.permute(1,0,2)

        # layer two
        h0 = torch.zeros( x.size(0), self.hidden_dim).requires_grad_() # one-directional
        h0=h0.to(device)
        hidden_state_list=[]
        for i , input_t in enumerate( hidden_state_list_1.chunk( hidden_state_list_1.size(1), dim=1 )):
            input_t=input_t.squeeze(1)
            h0 = self.GRU_Cell_forward_4(input_t, h0)
            hidden_state_list += [h0]
        hidden_state_list = torch.stack(hidden_state_list, 0)
        hidden_state_list = hidden_state_list.permute(1,0,2)

        hidden_state_list = self.input_LN_forward_2(hidden_state_list)
        attn_weight_matrix_forward_S1 = self.attention_net_2( hidden_state_list )
        hidden_matrix_forward_S1 = torch.bmm( attn_weight_matrix_forward_S1, hidden_state_list )

        out_forward_S1 = torch.relu( self.fc_layer_2( hidden_matrix_forward_S1.view( -1 , hidden_matrix_forward_S1.size()[1]*hidden_matrix_forward_S1.size()[2] ) ) )

        out = self.label( torch.cat( (out_forward_M1, out_forward_S1), 1 ) )

        return out, attn_weight_matrix_forward_M1, attn_weight_matrix_forward_S1