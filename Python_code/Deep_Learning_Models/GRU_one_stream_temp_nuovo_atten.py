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
        # self.input_LN_forward = torch.nn.LayerNorm( [max_timestep, hidden_dim], elementwise_affine=True)

        # for CNN
        '''
        self.conv1 = torch.nn.Conv2d( 1, 1, ( 1, hidden_dim), 1 , padding=0)
        self.conv2 = torch.nn.Conv2d( 1, 1, ( 2, hidden_dim), 1 , padding=0)
        self.conv3 = torch.nn.Conv2d( 1, 1, ( 3, hidden_dim), 1 , padding=0)
        '''


        r = int( max_timestep/2 )
        da= int( hidden_dim/2 )

        self.W_s1_1 = torch.nn.Linear( hidden_dim, da )
        self.W_s2_1 = torch.nn.Linear( da, r )

        # LN inside atten
        self.LN_in_atten = torch.nn.LayerNorm([max_timestep, da], elementwise_affine=False)

        # big
        '''
        self.fc_layer_x = torch.nn.Linear( hidden_dim*max_timestep, int( (hidden_dim*max_timestep)/2) )
        self.label_x = torch.nn.Linear( int( (hidden_dim*max_timestep)/2), 1 )

        self.fc_layer_y = torch.nn.Linear( hidden_dim*max_timestep, int( (hidden_dim*max_timestep)/2))
        self.label_y = torch.nn.Linear( int( (hidden_dim*max_timestep)/2), 1 )
        '''

        # two stage
        self.x_stage_1 = torch.nn.Linear( hidden_dim,  int(max_timestep/2) )
        self.x_stage_2 = torch.nn.Linear( int(max_timestep*(max_timestep/2)), 1)

        self.y_stage_1 = torch.nn.Linear( hidden_dim,  int(max_timestep/2) )
        self.y_stage_2 = torch.nn.Linear( int(max_timestep*(max_timestep/2)), 1)

        # max or sum
        '''
        self.fc_layer_x = torch.nn.Linear( max_timestep, int( (max_timestep)/2) )
        self.label_x = torch.nn.Linear( int( (max_timestep)/2), 1 )

        self.fc_layer_y = torch.nn.Linear( max_timestep, int( (max_timestep)/2))
        self.label_y = torch.nn.Linear( int( (max_timestep)/2), 1 )
        '''

        # for CNN
        '''
        self.fc_layer_x = torch.nn.Linear( int(3*max_timestep-3), int( (max_timestep)/2) )
        self.label_x = torch.nn.Linear( int( (max_timestep)/2), 1 )

        self.fc_layer_y = torch.nn.Linear( int(3*max_timestep-3), int( (max_timestep)/2))
        self.label_y = torch.nn.Linear( int( (max_timestep)/2), 1 )
        '''

    def attention_net_1(self, gru_output):

        # LN inside atten
        attn_weight_matrix = self.W_s2_1( torch.tanh( self.LN_in_atten(self.W_s1_1(gru_output))) )
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        attn_weight_matrix = torch.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix

    def attention_map_conv(self, attn_weight_matrix, hidden_matrix ):
        
        result_for_all_r = []
        # how many r's
        for i in range( attn_weight_matrix.size()[1] ):          
            the_attention_vector = torch.transpose( attn_weight_matrix[:,i,:].unsqueeze(1) , 1, 2)

            result_for_one_r = []
            # how many units to be conducted temp atten            
            for j in range( hidden_matrix.size()[2] ):
                the_vector = hidden_matrix[:,:,j].unsqueeze(2)
                output = torch.mul( the_attention_vector, the_vector )
                result_for_one_r += [output]

            result_for_one_r = torch.stack(result_for_one_r, 0) # (num of GRU units, batch, time step, 1)
            result_for_one_r = result_for_one_r.permute(3,1,2,0)
            result_for_one_r = result_for_one_r.squeeze(0)

            result_for_all_r += [result_for_one_r]

        result_for_all_r = torch.stack(result_for_all_r, 0)
        result_for_all_r = result_for_all_r.permute(1,0,2,3)

        result_for_all_r = torch.sum(result_for_all_r, 1, keepdim = True)

        # result_for_all_r = torch.max(result_for_all_r, 1, keepdim = True) # Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim
        # result_for_all_r = result_for_all_r[0]

        return result_for_all_r

    def attention_map_and_hidden_states_mul(self, attn_weight_matrix, hidden_matrix ):

        result_for_all_r = []
        # how many r's
        for i in range( attn_weight_matrix.size()[1] ):
            the_attention_vector = torch.transpose( attn_weight_matrix[:,i,:].unsqueeze(1) , 1, 2)
            # print('size of the_attention_vector = ', the_attention_vector.size(), '\n')
            the_attention_matrix = the_attention_vector.expand( hidden_matrix.size() )
            # print('size of the_attention_matrix = ', the_attention_matrix.size(), '\n')
            # print('size of hidden_matrix = ', hidden_matrix.size(), '\n')
            result_for_one_r = torch.mul(the_attention_matrix, hidden_matrix)
            # print('size of result_for_one_r = ', result_for_one_r.size(), '\n')
            result_for_all_r += [result_for_one_r]
        result_for_all_r = torch.stack(result_for_all_r, 0)
        # print('size of result_for_all_r 1 = ', result_for_all_r.size(), '\n')
        result_for_all_r = result_for_all_r.permute(1,0,2,3)
        # print('size of result_for_all_r 2 = ', result_for_all_r.size(), '\n')
        result_for_all_r = torch.sum(result_for_all_r, 1, keepdim = True)
        # result_for_all_r = result_for_all_r.squeeze(1)
        # print('size of result_for_all_r = ', result_for_all_r.size(), '\n')

        return result_for_all_r # size (batch, 1, time step, hidden units)

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

        # hidden_state_list = self.input_LN_forward(hidden_state_list)

        attn_weight_matrix_forward = self.attention_net_1( hidden_state_list )
        

        # hidden_matrix_forward = torch.bmm( attn_weight_matrix_forward, hidden_state_list )
        # feature_map_after_atten = self.attention_map_conv(attn_weight_matrix_forward, hidden_state_list )        
        feature_map_after_atten = self.attention_map_and_hidden_states_mul(attn_weight_matrix_forward, hidden_state_list )


        # original
        '''
        hidden_matrix_forward_x = feature_map_after_atten.clone()
        hidden_matrix_forward_y = feature_map_after_atten.clone()

        out_x = torch.relu( self.fc_layer_x( hidden_matrix_forward_x.view( -1 , hidden_matrix_forward_x.size()[1]*hidden_matrix_forward_x.size()[2]*hidden_matrix_forward_x.size()[3] ) ) )
        out_y = torch.relu( self.fc_layer_y( hidden_matrix_forward_y.view( -1 , hidden_matrix_forward_y.size()[1]*hidden_matrix_forward_y.size()[2]*hidden_matrix_forward_x.size()[3] ) ) )

        out_x = self.label_x( out_x )
        out_y = self.label_y( out_y )

        out = torch.cat(  (out_x, out_y),1 )
        '''
        # two stage
        out_x = torch.relu(self.x_stage_1( feature_map_after_atten.clone() ) )
        out_y = torch.relu(self.y_stage_1( feature_map_after_atten.clone() ) )

        out_x = out_x.view(-1, out_x.size()[1]*out_x.size()[2]*out_x.size()[3] )
        out_y = out_y.view(-1, out_y.size()[1]*out_y.size()[2]*out_y.size()[3] )

        out_x = self.x_stage_2(out_x)
        out_y = self.y_stage_2(out_y)

        out = torch.cat(  (out_x, out_y),1 )

        # max or sum
        '''
        feature_map_after_atten = torch.sum(feature_map_after_atten, 3, keepdim = True)

        # feature_map_after_atten = torch.max(feature_map_after_atten, 3, keepdim = True)
        # feature_map_after_atten = feature_map_after_atten[0]

        hidden_matrix_forward_x = feature_map_after_atten.clone()
        hidden_matrix_forward_y = feature_map_after_atten.clone()

        out_x = ( self.fc_layer_x( hidden_matrix_forward_x.view( -1 , hidden_matrix_forward_x.size()[1]*hidden_matrix_forward_x.size()[2]*hidden_matrix_forward_x.size()[3] ) ) )
        out_y = ( self.fc_layer_y( hidden_matrix_forward_y.view( -1 , hidden_matrix_forward_y.size()[1]*hidden_matrix_forward_y.size()[2]*hidden_matrix_forward_x.size()[3] ) ) )

        out_x = self.label_x( out_x )
        out_y = self.label_y( out_y )

        out = torch.cat(  (out_x, out_y),1 )
        '''

        # two stage


        # for CNN
        '''
        result_conv1 = self.conv1(feature_map_after_atten)
        result_conv2 = self.conv2(feature_map_after_atten)
        result_conv3 = self.conv3(feature_map_after_atten)

        result_conv1 = result_conv1.view(-1, result_conv1.size()[1]*result_conv1.size()[2]*result_conv1.size()[3] )
        result_conv2 = result_conv2.view(-1, result_conv2.size()[1]*result_conv2.size()[2]*result_conv2.size()[3] )
        result_conv3 = result_conv3.view(-1, result_conv3.size()[1]*result_conv3.size()[2]*result_conv3.size()[3] )

        result = torch.cat( (result_conv1, result_conv2, result_conv3) ,1)

        hidden_matrix_forward_x = result.clone()
        hidden_matrix_forward_y = result.clone()

        out_x = torch.relu( self.fc_layer_x( hidden_matrix_forward_x ))
        out_y = torch.relu( self.fc_layer_y( hidden_matrix_forward_y ))

        out_x = self.label_x( out_x )
        out_y = self.label_y( out_y )

        out = torch.cat(  (out_x, out_y),1 )
        '''

        return out, attn_weight_matrix_forward
