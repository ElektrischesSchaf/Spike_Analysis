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

        r = int( max_timestep/4 )
        da= int( hidden_dim/2 )

        self.W_s1_1 = torch.nn.Linear( 2*hidden_dim, da )
        self.W_s2_1 = torch.nn.Linear( da, r )

        self.fc_layer_x = torch.nn.Linear( 2*r*hidden_dim, int(hidden_dim))
        self.label_x = torch.nn.Linear( int(hidden_dim), 1 )

        self.fc_layer_y = torch.nn.Linear( 2*r*hidden_dim, int(hidden_dim))
        self.label_y = torch.nn.Linear( int(hidden_dim), 1 )

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
        hidden_matrix = torch.bmm( attn_weight_matrix, hidden_state_list_2_result )

        hidden_matrix_x = hidden_matrix.clone()
        hidden_matrix_y = hidden_matrix.clone()

        out_x = torch.relu( self.fc_layer_x( hidden_matrix_x.view( -1 , hidden_matrix_x.size()[1]*hidden_matrix_x.size()[2] ) ) )
        out_y = torch.relu( self.fc_layer_x( hidden_matrix_y.view( -1 , hidden_matrix_y.size()[1]*hidden_matrix_y.size()[2] ) ) )

        out_x = self.label_x( out_x )
        out_y = self.label_y( out_y )

        out = torch.cat( (out_x, out_y) ,1)

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

        # LN inside atten
        # self.LN_in_atten = torch.nn.LayerNorm([max_timestep, da], elementwise_affine=False)

        self.fc_layer_x = torch.nn.Linear( hidden_dim*max_timestep, int( (hidden_dim*max_timestep)/2) )
        self.label_x = torch.nn.Linear( int( (hidden_dim*max_timestep)/2), 1 )

        self.fc_layer_y = torch.nn.Linear( hidden_dim*max_timestep, int( (hidden_dim*max_timestep)/2))
        self.label_y = torch.nn.Linear( int( (hidden_dim*max_timestep)/2), 1 )

    def attention_net_1(self, gru_output):

        # LN inside atten
        # attn_weight_matrix = self.W_s2_1( torch.tanh( self.LN_in_atten( self.W_s1_1(gru_output) )  ))

        attn_weight_matrix = self.W_s2_1( torch.tanh( self.W_s1_1(gru_output) ) )
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        attn_weight_matrix = torch.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix

    def attention_map_conv(self, attn_weight_matrix, hidden_matrix ):
        
        result_for_all_r = []
        # how many r's
        for i in range( attn_weight_matrix.size()[1] ):
            # print('shape attn_weight_matrix = ', attn_weight_matrix.size(), '\n')
            
            the_attention_vector = torch.transpose( attn_weight_matrix[:,i,:].unsqueeze(1) , 1, 2)

            result_for_one_r = []
            # how many units to be conducted temp atten            
            for j in range( hidden_matrix.size()[2] ):
                the_vector = hidden_matrix[:,:,j].unsqueeze(2)
                # print(the_attention_vector.size(), ' ', the_vector.size(), '\n')
                output = torch.mul( the_attention_vector, the_vector )
                result_for_one_r += [output]

            result_for_one_r = torch.stack(result_for_one_r, 0) # (num of GRU units, batch, time step, 1)
            result_for_one_r = result_for_one_r.permute(3,1,2,0)
            result_for_one_r = result_for_one_r.squeeze(0)
            # print('size of result_for_one_r = ', result_for_one_r.size(), '\n')

            result_for_all_r += [result_for_one_r]

        result_for_all_r = torch.stack(result_for_all_r, 0)
        result_for_all_r = result_for_all_r.permute(1,0,2,3)
        

        result_for_all_r = torch.sum(result_for_all_r, 1, keepdim = True)
        # print('size of result_for_all_r = ', result_for_all_r.size(), '\n')

        return result_for_all_r

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
        

        # hidden_matrix_forward = torch.bmm( attn_weight_matrix_forward, hidden_state_list )
        feature_map_after_atten = self.attention_map_conv(attn_weight_matrix_forward, hidden_state_list )

        hidden_matrix_forward_x = feature_map_after_atten.clone()
        hidden_matrix_forward_y = feature_map_after_atten.clone()

        out_x = torch.relu( self.fc_layer_x( hidden_matrix_forward_x.view( -1 , hidden_matrix_forward_x.size()[1]*hidden_matrix_forward_x.size()[2]*hidden_matrix_forward_x.size()[3] ) ) )
        out_y = torch.relu( self.fc_layer_y( hidden_matrix_forward_y.view( -1 , hidden_matrix_forward_y.size()[1]*hidden_matrix_forward_y.size()[2]*hidden_matrix_forward_x.size()[3] ) ) )

        out_x = self.label_x( out_x )
        out_y = self.label_y( out_y )

        out = torch.cat(  (out_x, out_y),1 )

        return out, attn_weight_matrix_forward
