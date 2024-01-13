import torch
import torch.nn as nn
import torch.optim as optim
from pose_output import *
from transformer_functions import *
class cnnLayer(nn.Module):
    def __init__(self):
        super(cnnLayer, self).__init__()
       
        self.cnn1 = nn.Conv2d(3, 32, kernel_size=(5, 7), stride = 2, padding=2)
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=(5,7), stride = 2, padding=2)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.cnn3 = nn.Conv2d(64, 32, 3, stride = 2, padding=2)
        self.cnn4 = nn.Conv2d(32, 16, 3, stride = 2)

        self.linear = [nn.Linear(2400, 64) for i in range(64)]
        
    def forward(self, input):
        cnn1 = self.cnn1(input)
        cnn2 = self.cnn2(cnn1)

        batch1 = self.batchnorm1(cnn2)
        cnn3 = self.cnn3(batch1)
        cnn4 = self.cnn4(cnn3)
        cnn4 = cnn4.permute(0,2,3,1)
        
        patches_tensor = create_embeddings(self.linear, cnn4)
        
        return patches_tensor

class PositionalEncoding(nn.Module):
    def __init__(self, D, maxL):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(positional_encoding_function(maxL, D).unsqueeze(0), requires_grad=False)
    def forward(self, x):
        # print("hhhhhhhhhh", x)
        return x  + self.pe[:,:x.size(1)]


class FeedForward(nn.Module):
    def __init__(self, D, D_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(D,D_ff) 
        self.fc2 = nn.Linear(D_ff, D) 
        self.relu = nn.ReLU() 

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x) 



class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super(MultiHeadAttention, self).__init__()
        assert D % H == 0, "D must be divisible by H"
        self.D = D
        self.H = H
        self.Dh = D//H
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
        self.W_v = nn.Linear(D, D)
        self.W_o = nn.Linear(D, D)
    def forward(self, Q, K, V, mask=None):
        Q = split_heads(self.W_q(Q), self.H)
        K = split_heads(self.W_k(K), self.H)
        V = split_heads(self.W_v(V), self.H)

        output = scaled_dot_product_attention(Q, K, V, self.Dh, mask, negative_infinity=-1e9)
        output = combine_heads(output)

        output = self.W_o(output)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, D, H, D_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(D,H) 
        self.feed_forward =  FeedForward(D, D_ff)
        self.norm1 = nn.LayerNorm(D) 
        self.norm2 = nn.LayerNorm(D) 
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, mask=None):
        attention = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention))
        ff = self.feed_forward(x)

        return self.norm2(x + self.dropout(ff)) 

class Transformer(nn.Module):
    def __init__(self, D,
                H, Nx, D_ff, maxL, dropout):
        super(Transformer, self).__init__()
        self.cnn = cnnLayer()
        self.positional_encoding = PositionalEncoding(D, maxL) 
        self.encoder_layers = nn.ModuleList([EncoderLayer(D, H, D_ff, dropout) for _ in range(Nx)]) 
        self.fc = nn.Linear(D, 2) 
        self.dropout = nn.Dropout(dropout) 
        self.outLay = outputLayer()

    def forward(self, source):
        cnn = self.cnn(source)
        source = self.positional_encoding(cnn)
        print(source.shape)
        encoder_output = source
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output)
        output = self.fc(encoder_output)
        f = output.view(output.shape[0], -1)
        output = self.outLay(f)
        return output

