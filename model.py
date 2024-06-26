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
    
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self,d_model:int,h:int,dropout:float) -> None:
        
        super().__init__()
        self.d_model = d_model
        self.h = h 
        assert d_model% h ==0 , "d_model is not divisible by h"
        
        self.d_k = d_model//h 
        self.w_q = nn.Linear(d_model,d_model) #Wq 
        self.w_k = nn.Linear(d_model,d_model) # Wk 
        self.w_v = nn.Linear (d_model,d_model) #Wv 
        
        self.w_o = nn.Linear(d_model,d_model) # Wo 
        self.dropout = nn.Dropout(dropout)

    @staticmethod 
    def attention(query,key,value,mask,droput:nn.Dropout):
        d_k = query.shape[-1]

        #(batch_size,seq_len,d_k) -> (batch_size,seq_len,seq_len)
        attention_score = (query @ key.tranpose(-2,-1) ) / math.sqrt(d_k)

        if mask is not None:
            attention_score.masked_fill(mask==0,-1e9)
        attention_score = attention_score.softmax(dim=-1) #(batch,seq_len,seq_len)
        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score@ value),attention_score



    def forward(self,q,k,v,mask):

        query = self.w_q(q) # (batch,seq_len,d_model0 -->(batch,seq_len,d_model)
        key = self.w_k(k)# (batch,seq_len,d_model0 -->(batch,seq_len,d_model)
        value = self.w_v(v)# (batch,seq_len,d_model0 -->(batch,seq_len,d_model) 


        # (batch-size,seq_len,d_model) --> (batch-size,seq_len,h,d_k) -> (batch_size,sh,seq_len,d_k)
        # we doing the transpose here becuase we want eveyr head to see the entire sequence of length but with diffent d_k 
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).tranpose(1,2)
        key  = key.view(key.shape(0),key.shape[1],self.h,self.d_k).tranpose(1,2)
        value = value.view(value.shape[0],value.shape[1]self.h,self.d_k).tranpose(1,2)

        x,self.attention_score = MultiHeadAttentionBlock(query,key,value,mask,self.dropout)
















        
        