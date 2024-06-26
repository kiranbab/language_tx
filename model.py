import torch 
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
        
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEmbedding(nn.Module):
    def __init__(self,d_model:int,seq_len:int, dropout:float)->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len  = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len,d_model)
        pe = torch.zeros(seq_len,d_model)
        # Create a vector of shape(seq_len,1)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        # Create a vector of shape (d_model)
        div_term = torch.arange(0,d_model,2).float() * (-math.log(10000)/d_model)
        
        # Apply sine to even indices 
        pe [:,0::2] = torch.sin (position * div_term) #sin(position * (10000 ** (2i/d_model))
        # apply cose to odd indices 
        pe[:,1::2] = torch.cos(position * div_term) #cos(position *(10000 ** (2i/d_model)))
        # add a batch dimension to poistional embedding 
        pe = pe.unsqueeze(0) #(1,seq_len,d_model)
        # register the poistional encoding as a bugger 
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        x = x+(self.pe[:,:x.shape[1] , :]).requires_grad(False) 
        return self.droput(x)
    
    
class LayerNormalization(nn.Module):
    def __init__(self,features:int,eps: float=10**-6) -> None:
        
        super().__init__()
        self.eps =eps
        self.alpha = nn.Parameter(torch.ones(features))#alpha is a learnabale parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnbale parameter
        
    def forward(self,x):
        # x : (batch,seq_len,hidden_size)
        # keep the dimension for broadcasting 
        mean  = x.mean (dim =-1, keepdim = True)# (batch,seq_len,1)
        # keep the dimesnion for broadcasting 
        std =x.std(dim=-1,keepdim =True)# (batch,seq_len,1)
        # eps is to prevent diving by zero or when std is very small 
        return self.alpha * (x-mean)/(std+self.eps) +self.bias
    
class FeedForwardBlock(nn.Module):
    
    def __init__(self,d_model:int,d_ff:int,dropout:float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)# w1 and b1 
        self.dropout= nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model) # w2 and b2 
    
    def forward(self,x):
        # (batch,seq_len,d_model ) -->(batch,seq_len,d_ff) --> (batch,seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    
    
        
        
        