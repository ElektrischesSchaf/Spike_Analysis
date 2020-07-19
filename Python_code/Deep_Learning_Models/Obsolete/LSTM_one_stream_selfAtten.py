# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class  LSTMModel(torch.nn.Module):

    def __init__(self, input_dim, batch_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        self.batch_dim=batch_dim
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
        
        # Readout layer
        # self.fc1 = torch.nn.Linear(hidden_dim, int(hidden_dim/2)) # one-directional
        # self.fc2 = torch.nn.Linear(int(hidden_dim/2), output_dim) # one-directional

        r = 30
        da= 350

        self.fc_layer = torch.nn.Linear( r*hidden_dim, int(hidden_dim/2))
        self.label = torch.nn.Linear( int(hidden_dim/2), batch_dim )

        self.W_s1 = torch.nn.Linear(hidden_dim, da)
        self.W_s2 = torch.nn.Linear(da, r)

    def attention_net(self, lstm_output):
        # lstm_output=lstm_output.permute(0, 2, 1)
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        # print('shape of attn_weight_matrix= ', attn_weight_matrix.size())
        attn_weight_matrix = torch.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, x):

        # x torch.Size([64, 96])
        time_step_size=int(x.size()[0])

        x=x.unsqueeze(0)       
        # m=torch.nn.LayerNorm( x.size()[:], elementwise_affine=False )
        # x=m(x)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_() # one-directional
        # h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_() # bidirectional
        h0=h0.to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_() # one-directional
        # c0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_() # bidirectional
        c0=c0.to(device)

        # print('input dim= ', x.size(), '\n') # input dim=  torch.Size([1, 64, 96]) => batch_first=True, (batch_dim, seq_dim, feature_dim)

        # time steps
        out, (hn, cn) = self.lstm(x, (h0,c0))
        # out size = (1, 64, hidden_dim)

        attn_weight_matrix = self.attention_net(out)

        hidden_matrix = torch.bmm(attn_weight_matrix, out)

        # print('hidden_matrix size=', hidden_matrix.size(), '\n')

        
        out = (self.fc_layer( hidden_matrix.view( -1 , hidden_matrix.size()[1]*hidden_matrix.size()[2] ) ) )
        out = self.label(out)
        out=out.view(time_step_size, -1)
        # print('out size= ', out.size(), '\n')

        # out=out.squeeze(0)
        # out size = (64, 96)
        
        return out
