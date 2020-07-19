# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropput):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = torch.nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=False)
        self.dropput = torch.nn.Dropout(dropput)

    def forward(self, x):

        x=x.unsqueeze(0)
        outputs, hidden = self.rnn(x)

        return hidden

class Decoder(torch.nn.Module):
    def __init__(self,  input_dim, output_dim, hidden_dim, n_layers, dropput):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = torch.nn.GRU(output_dim, hidden_dim, n_layers, batch_first=True, bidirectional=False)
        self.dropput = torch.nn.Dropout(dropput)

        self.fc1 = torch.nn.Linear(hidden_dim, int(hidden_dim/2)) # one-directional
        self.fc2 = torch.nn.Linear(int(hidden_dim/2), output_dim) # one-directional

    def forward(self, input, hidden):

        input=input.unsqueeze(0)
        input=input.unsqueeze(0)

        # print('yee= ', input.size() )
        # input=self.dropput(input)
        # print('type of input= ', type(input))
        output, hidden = self.rnn( input , hidden )

        out = torch.relu(self.fc1(output))
        out = self.fc2(out)
        out=out.squeeze(0)

        return out, hidden

class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder=encoder
        self.decoder=decoder
        self.device=device

        assert encoder.hidden_dim==decoder.hidden_dim, \
        'Hidden dimensions of encoder and decoder must be equal!'
        assert encoder.n_layers==decoder.n_layers,\
        'Encoder and decoders must have equal number of layers!'
    
    def forward(self, input_signal, target, teacher_forcing_ratio=0.5):

        # input_signal=[batch_size, 96]
        # target=[batch_size, 1]

        target_len=target.size(0)
        outputs=torch.zeros( target.size(0), target.size(1) ).to(self.device)
        hidden=self.encoder(input_signal)

        input=target[0,:]
        # print('yee2 = ', input.size(), '\n\n\n\n')
        for t in range( target_len ):

            output, hidden = self.decoder( input , hidden)

            outputs[t,:] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1=output.squeeze(0)

            # print('top1= ', top1.size(),'  ')
            # print('target[t,:]= ', target[t,:].size(), '\n')

            # if teacher_force:
            #     input=target[t,:].float()   
            # else:
            #     input=top1.float()


            if teacher_forcing_ratio==0:
                input=top1
            else:
                input=target[t,:].float()

        return outputs

'''
class  GRUModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
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

        # x torch.Size([64, 96])

        x=x.unsqueeze(0)       
        # m=torch.nn.LayerNorm( x.size()[:], elementwise_affine=False )
        # x=m(x)

        # print('input dim= ', x.size(), '\n') # input dim=  torch.Size([1, 64, 96]) => batch_first=True, (batch_dim, seq_dim, feature_dim)

        # time steps
        out, _ = self.GRU(x)

        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        out=out.squeeze(0)

        return out
'''