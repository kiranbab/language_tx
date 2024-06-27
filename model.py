import torch 
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    """
    A module to create and apply an embedding layer for input sequences.

    Parameters:
    - d_model (int): The dimensionality of the output embedding vectors.
    - vocab_size (int): The size of the vocabulary, i.e., the number of unique tokens that can be embedded.

    Methods:
    - forward(x): Applies the embedding layer to the input tensor `x` and scales it by the square root of `d_model`.

    Inputs:
    - x (Tensor): A tensor of token indices with shape `(batch_size, sequence_length)`.

    Returns:
    - Tensor: The embedded input tensor scaled by the square root of `d_model`, with shape `(batch_size, sequence_length, d_model)`.

    Raises:
    - No explicit exceptions are raised but expect errors related to incompatible tensor shapes or types.
    """
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    """
    A module to create and apply positional embeddings to input sequences.

    Parameters:
    - d_model (int): The dimensionality of the output vectors.
    - seq_len (int): The maximum length of the input sequences.
    - dropout (float): The dropout rate to apply to the output of this layer.

    Attributes:
    - d_model: Stores the dimensionality of the output vectors.
    - seq_len: Stores the maximum length of the input sequences.
    - dropout: A dropout layer with the specified dropout rate.
    - pe: A positional encoding tensor with shape (1, seq_len, d_model).

    Methods:
    - forward(x): Adds positional encodings to the input tensor `x`.

    Inputs:
    - x (Tensor): A tensor of shape `(batch_size, sequence_length, d_model)` representing the input sequences.

    Returns:
    - Tensor: The input tensor with added positional encodings, after dropout has been applied, with shape `(batch_size, sequence_length, d_model)`.

    Raises:
    - No explicit exceptions are raised but expect errors related to incompatible tensor shapes or types.
    """
    def __init__(self, d_model:int, seq_len:int, dropout:float)->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len  = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Create a vector of shape (d_model)
        div_term = torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
        
        # Apply sine to even indices 
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i/d_model)))
        # Apply cosine to odd indices 
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i/d_model)))
        # Add a batch dimension to positional embedding 
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer 
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


    
class LayerNormalization(nn.Module):
    """
    Implements a Layer Normalization module as described in the paper "Layer Normalization" by Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton.

    Parameters:
    - features (int): The number of features in the input tensor. It is the size of the last dimension of the input tensor.
    - eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
    - eps: The epsilon value for numerical stability in normalization.
    - alpha: A learnable scaling parameter for the normalized tensor. Shape is (features,).
    - bias: A learnable bias parameter to be added to the normalized tensor. Shape is (features,).

    Methods:
    - forward(x): Normalizes the input tensor `x`, scales it with `alpha`, and shifts it with `bias`.

    Inputs:
    - x (Tensor): The input tensor with shape `(batch, seq_len, features)`.

    Returns:
    - Tensor: The normalized, scaled, and shifted tensor with the same shape as input `x`.

    Raises:
    - No explicit exceptions are raised but expect errors related to incompatible tensor shapes or types.
    """
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


    
class FeedForwardBlock(nn.Module):
    """
    A feedforward neural network block as part of a Transformer model architecture.

    Parameters:
    - d_model (int): The dimensionality of the input and output tensors.
    - d_ff (int): The dimensionality of the hidden layer.
    - dropout (float): The dropout rate applied to the output of the first linear transformation.

    Methods:
    - forward(x): Applies two linear transformations with a ReLU activation and dropout in between.

    Inputs:
    - x (Tensor): The input tensor with shape `(batch_size, sequence_length, d_model)`.

    Returns:
    - Tensor: The output tensor with the same shape as the input tensor `(batch_size, sequence_length, d_model)`.

    Raises:
    - No explicit exceptions are raised but expect errors related to incompatible tensor shapes or types.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    """
    Implements a Multi-Head Attention mechanism as part of the Transformer model architecture.

    Parameters:
    - d_model (int): The dimensionality of the input and output tensors.
    - h (int): The number of attention heads.
    - dropout (float): The dropout rate applied to the attention scores.

    Attributes:
    - d_k (int): The dimensionality of the keys, queries, and values in each attention head.
    - w_q, w_k, w_v, w_o (nn.Linear): Linear layers for queries, keys, values, and output projection.
    - dropout (nn.Dropout): Dropout layer applied to the attention scores.

    Methods:
    - attention(query, key, value, mask, dropout): Static method that computes the attention scores and the weighted sum of values.
    - forward(q, k, v, mask): Computes the output of the Multi-Head Attention block for input queries, keys, and values.

    Inputs:
    - q, k, v (Tensor): Tensors containing queries, keys, and values. They have shape `(batch_size, sequence_length, d_model)`.
    - mask (Tensor, optional): A mask tensor with shape `(batch_size, sequence_length, sequence_length)` to prevent attention to certain positions.

    Returns:
    - Tensor: The output tensor of the Multi-Head Attention block with shape `(batch_size, sequence_length, d_model)`.

    Raises:
    - AssertionError: If `d_model` is not divisible by `h`, indicating a configuration error in the attention heads.
    """
    
    
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

        # (batch_size,h,seq_len,d_k) -> (batch_size,seq_len,h,d_k) -> (batch_size,seq_len,d_model)
        x =x.tranpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)

        # (batch_size,seq_len,d_model) -> (batch_size,seq_len,d_model)
        return self.w_o(x) 


class ResidualConnection(nn.Module):
    """
    Implements a residual connection followed by a dropout layer.

    This module is designed to be used in a Transformer model, where it can help mitigate the vanishing gradient problem and improve training stability.

    Parameters:
    - dropout (float): The dropout rate to apply after the sublayer operation.

    Methods:
    - forward(x, sublayer): Applies layer normalization to `x`, passes it through a sublayer (with arbitrary functionality), applies dropout to the sublayer's output, and adds this result to the original `x` for the residual connection.

    Inputs:
    - x (Tensor): The input tensor to which the residual connection will be applied. Shape is typically `(batch_size, sequence_length, d_model)` where `d_model` is the dimensionality of the input.
    - sublayer (callable): A function or module that takes the normalized `x` as input and returns a tensor of the same shape. This represents the operation to be applied to `x` before adding the residual connection.

    Returns:
    - Tensor: The output tensor after applying the residual connection and dropout. Has the same shape as `x`.

    Raises:
    - No explicit exceptions are raised by this module, but errors may occur if `sublayer` does not return a tensor of the same shape as its input.
    """
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization() 

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    """
    A Transformer Encoder Block that combines a self-attention mechanism with a feed-forward neural network.

    Parameters:
    - self_attention_block (MultiHeadAttentionBlock): The self-attention mechanism block.
    - feed_forward_block (FeedForwardBlock): The feed-forward neural network block.
    - dropout (float): The dropout rate used in the residual connections.

    Methods:
    - forward(x, src_mask): Processes the input tensor `x` using the self-attention mechanism and the feed-forward network within the context of the provided source mask `src_mask`.

    Inputs:
    - x (Tensor): The input tensor to the encoder block with shape `(batch_size, sequence_length, d_model)`.
    - src_mask (Tensor): The mask tensor for the source input with shape `(batch_size, sequence_length, sequence_length)` used in the self-attention mechanism to prevent attention to certain positions.

    Returns:
    - Tensor: The output tensor of the encoder block with the same shape as the input tensor `(batch_size, sequence_length, d_model)`.

    Raises:
    - No explicit exceptions are raised by this method, but errors may occur due to incorrect input shapes or types.
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
class Encoder(nn.Module):
    """
    A Transformer Encoder module that sequentially applies a list of encoder layers and a final layer normalization.

    Parameters:
    - layers (nn.ModuleList): A list of encoder layers (instances of EncoderBlock or similar) to be applied sequentially.

    Methods:
    - forward(x, mask): Processes the input tensor `x` through each encoder layer using the mask `mask`, followed by layer normalization.

    Inputs:
    - x (Tensor): The input tensor to the encoder with shape `(batch_size, sequence_length, d_model)`.
    - mask (Tensor): The mask tensor for the input with shape `(batch_size, sequence_length, sequence_length)` used in self-attention mechanisms to prevent attention to certain positions.

    Returns:
    - Tensor: The output tensor of the encoder with the same shape as the input tensor `(batch_size, sequence_length, d_model)`.

    Raises:
    - No explicit exceptions are raised by this method, but errors may occur due to incorrect input shapes or types.
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# all the code use above is for Encoder block of Transformer Architecture 
# ________________________________________________________________________________________________________
# DEFINING DECODER: 

class DecoderBlock(nn.Module):
    
    def __init__(self,) -> None:
        super().__init__()















        
        