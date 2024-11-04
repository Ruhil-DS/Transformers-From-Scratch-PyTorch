import torch
import torch.nn as nn
import math

# First component of the transformer model - Input embeddings
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimension of the model.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # this was mentioned in the original paper - Attention is all you need
    

# Second component of the transformer model - Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length:int, dropout:float) -> None:
        """
        Args:
            d_model (int): The dimension of the model.
            seq_length (int): The length of the sequence.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        # PE is of length seq_length x d_model
        # This is because the maximum length of the sequence is seq_length and the dimension of the model is d_model
        pe = torch.zeros(seq_length, d_model)

        # create a vector of shape (seq_length, 1)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        
        # div term will be calculated in log space for numerical stability
        # each single term will be used for both sin and cos; hence, we take a step size of 2
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (- math.log(10000.0) / d_model))

        # sin is used for even indices and cos is used for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_length, d_model)

        self.register_buffer('pe', pe) # this is a buffer that is registered in the model but not as a parameter

    def forward(self, x):
        # we don't need to compute gradients for PE, hence we set requires_grad to False
        x += (self.pe[:x.size(1), :]).requires_grad_(False)
        return self.dropout(x)
    

# Third component of the transformer model - Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        """
        Args:
            eps (float): A small constant to avoid division by zero.
        """
        super().__init__()
        self.eps = eps  # eps is a small constant to avoid division by zero
        self.alpha  = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # Add sqrt in the denominator
        return self.alpha * (x - mean) / torch.sqrt(std.pow(2) + self.eps) + self.bias
    


# Fourth component of the transformer model - Feed Forward Sublayer
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        """
        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feed forward network.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2
    
    def forward(self, x):
        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_ff) --> (batch_size, seq_length, d_model)
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
    

# Fifth component of the transformer model - Multi-Head Attention
class MultiHeadAttentionBlock(nn.Module):
    # out input is transformed into Q, K, V
    # Q: query, K: key, V: value
    # We multiply Q with W^Q, K with W^K, V with W^V
    # These are then split into multiple heads and processed in parallel
    # splitting is done along the embedding dimension (last dimension)
    # each head has its own W^Q, W^K, W^V
    # finally, the results of the different heads are concatenated and multiplied by W^O
    # The final result is a MH-Attention block
    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        """
        Args:
            d_model (int): The dimension of the model.
            h (int): The number of heads.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # dimension of the vector seen by each head

        self.w_q = nn.Linear(d_model, d_model) # W^Q
        self.w_k = nn.Linear(d_model, d_model) # W^K
        self.w_v = nn.Linear(d_model, d_model) # W^V
        
        self.w_o = nn.Linear(d_model, d_model) # W^O -> Here, h*d_v = d_model

    @staticmethod  # static method is used to define a method that is shared across all instances of the class
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1] # last dim of the query

        # (batch_size, h, seq_length, d_k) @ (batch_size, h, d_k, seq_length) --> (batch_size, h, seq_length, seq_length)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.
            v (torch.Tensor): The value tensor.
            mask (torch.Tensor, optional): The mask tensor. Defaults to None.
        """
        # mask is used to mask out the padding tokens
        # mask is of shape (batch_size, h, seq_length, seq_length)
        # mask is 0 for the padding tokens and 1 for the actual tokens
        # mask is used to ensure that the model does not attend to the padding tokens
        # basically, if we don't want some words to attend to other words, we can use a mask
        # in the original paper, mask is used to mask out the padding tokens
        
        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        query = self.w_q(q) # this is basically multiplying Q with W^Q
        key = self.w_k(k) # this is basically multiplying K with W^K
        value = self.w_v(v) # this is basically multiplying V with W^V

        # we are splitting the query into multiple heads
        # for eg, we can go from this to -->  from: (32, 50, 512) --> (32, 50, 8, 64)
        # transpose was added to achieve conventional shape for attention calculations
        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, h, d_k) --> (batch_size, h, seq_length, d_k)
        # we want each head to see (..., seq_length, d_k) part
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # now, the next part is calculating the attention using the scaled dot product
        # we use a new function defined (above) to calculate the attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_length, d_k) --> (batch_size, seq_length, h, d_k) --> (batch_size, seq_length, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.d_model)

        # finally, we multiply the result by W^O
        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        x = self.w_o(x)

        return x
    
# Sixth component of the transformer model - residual connection
# This is needed to ensure the skip connections are maintained
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        # return x + self.dropout(self.norm(sublayer(x))) # this is the original paper implementation


# Seventh component of the transformer model - Atomic Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # two residual connections are used in the encoder block as per the original paper
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask=None):
        """
        Args:
            x (torch.Tensor): The input tensor.
            src_mask (torch.Tensor, optional): The source mask tensor. Defaults to None.
        """
        # source mask is used to mask out the padding tokens in the source sentence
        # it means that we don't want the model to attend to the padding tokens
        # basically, if we don't want some words to attend to other words, we can use a mask
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    

# Eighth component of the transformer model - Encoder: This will have N atomic encoder blocks
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

# Ninth component of the transformer model - Atomic Decoder block
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # because we have 3 residual connections in the decoder part according to the original paper
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, lambda x: self.feed_forward_block)
        return x
    

# Tenth component of the transformer model - Decoder: This will have N atomic decoder blocks
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers - layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return self.norm(x)
    

# Eleventh component of the transformer model - Projection Layer (final layer in the decoder)
class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocal_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocal_size)

    def forward(self, x):
        # (batch, seq_length, d_model) --> (batch, seq_length, vocab_size)
        # we apply log_softmax to get the probabilities
        # we use log_softmax because it is numerically more stable
        return torch.log_softmax(self.proj(x), dim=-1)


# Twelfth component of the transformer model - Transformer main block
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, 
                 decoder: Decoder, 
                 src_embedding: InputEmbeddings, 
                 tgt_embedding: InputEmbeddings, 
                 src_pos: PositionalEncoding, 
                 tgt_pos: PositionalEncoding, 
                 projection_layer: ProjectionLayer) -> None:
        """
        Args:
            encoder (Encoder): The encoder.
            decoder (Decoder): The decoder.
            src_embedding (InputEmbeddings): The source embedding.
            tgt_embedding (InputEmbeddings): The target embedding.
            src_pos (PositionalEncoding): The source positional encoding.
            tgt_pos (PositionalEncoding): The target positional encoding.
            projection_layer (ProjectionLayer): The projection layer.
        """
                 
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def projection(self, x):
        return self.projection_layer(x)
    


def build_transfomer(src_vocab_size: int, 
                     tgt_vocab_size: int,
                     src_seq_length: int,
                     tgt_seq_length: int,
                     d_model: int = 512,
                     N: int = 6,
                     h: int = 8,
                     dropout: float = 0.1,
                     d_ff: int = 2048) -> None:
    """
    Args:
        src_vocab_size (int): The size of the source vocabulary.
        tgt_vocab_size (int): The size of the target vocabulary.
        src_seq_length (int): The length of the source sequence.
        tgt_seq_length (int): The length of the target sequence.
        d_model (int, optional): The dimension of the model. Defaults to 512.
        N (int, optional): The number of encoder and decoder blocks. Defaults to 6.
        h (int, optional): The number of heads. Defaults to 8.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        d_ff (int, optional): The dimension of the feed forward network. Defaults to 2048.
    """
    # embedding layers
    src_embedding = InputEmbeddings(src_vocab_size, d_model)
    tgt_embedding = InputEmbeddings(tgt_vocab_size, d_model)

    # position encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_length, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_length, dropout)

    # encoder/decoder-blocks
    encoder_blocks = []
    decoder_blocks = []
    for _ in range(N):
        # encoder
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

        # decoder
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # encoder/decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # projection layer
    # tgt_vocab sized is used because we are predicting the target sentence
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # transformer
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection_layer)

    # initialize the params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
