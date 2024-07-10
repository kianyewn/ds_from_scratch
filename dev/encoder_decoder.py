import torch.nn as nn
import torch
import torch.nn.functional as F

class GEGLU(nn.Module): 
    def forward(self, x:torch.tensor):
        x, gates = x.chunk(chunks=2, dim=-1)
        return x *  F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout=0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 2 * dim * mult),
                                GEGLU(),
                                nn.Linear(dim * mult, dim),
                                nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=dim)
    def forward(self, x,fn, **kwargs):
        return fn(self.norm(x), **kwargs)    

def invert_attention_mask(encoder_attention_mask):
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.

    Returns:
        `torch.Tensor`: The inverted attention mask.
    """

    dtype = encoder_attention_mask.dtype
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(dtype).min

    return encoder_extended_attention_mask


class Attention(nn.Module):
    def __init__(self,
                  dim, 
                  heads,
                  dim_head,
                  attn_dropout,
                  residual_dropout):
         super().__init__()
         self.inner_dim = heads * dim_head
         self.scale = dim_head ** -0.5
         self.heads = heads
         self.dim_head = dim_head
         
         self.query = nn.Linear(dim, self.inner_dim)
         self.key = nn.Linear(dim, self.inner_dim)
         self.value = nn.Linear(dim, self.inner_dim)
         self.to_out = nn.Linear(self.inner_dim, dim)
         
         self.attn_dropout = nn.Dropout(attn_dropout)
         self.resid_dropout = nn.Dropout(residual_dropout)
         
    def forward(self, query, key, value, attention_mask=None):
        """_summary_

        Args:
            query (_type_): (B, H, T, D)
            key (_type_): _description_
            value (_type_): _description_
            attention_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        B, T, D = query.shape
        q, k, v = self.query(query), self.key(key), self.value(value) # (B,T,D)  - > (B,T, inner_dim)
        q, k, v = (q.reshape(B, T, self.heads, self.dim_head).transpose(1,2),\
                k.reshape(B, T, self.heads, self.dim_head).transpose(1,2),\
                v.reshape(B, T, self.heads, self.dim_head).transpose(1,2),) # (B, T, inner_dim) -> (B,T,H,dim_head) -> (B, H, T, dim_head)
    
        sim = torch.matmul(q, k.transpose(-1,-2)) # (B, H, T, DH), (B, H, DH, T) -> (B, H, T ,T)
        sim = sim * self.scale
        # self.sim = sim 
        if attention_mask is not None:
            extended_attention_mask = invert_attention_mask(attention_mask)
            # self.extended_attention_mask = extended_attention_mask
            # print(extended_attention_mask.shape)
            sim = sim + extended_attention_mask  # positions with attention_mask == 0 has very large negative value

        sim = nn.functional.softmax(sim, dim=-1)
        attention = self.attn_dropout(sim)
        
        attention = torch.matmul(attention, v) # (B,H,T,T), (B, H,T, DH) -> (B,H,T,DH)
        attention = attention.transpose(1,2).flatten(start_dim=2) # (B,H,T,DH) -> (B,T,H,DH) -> (B,T, H * DH) 
        attention = self.to_out(attention) # (B,T, H*DH) -> (B,T, D)
        attention = self.resid_dropout(attention)
    
        return attention
       
class TransFormerEncoder(nn.Module):
    def __init__(self,
                 dim,
                 vocab_size,
                 depth,
                 heads,
                 dim_head,
                 attn_dropout=0,
                 residual_dropout=0,
                 ff_dropout=0,
                 ff_mult=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers =  nn.ModuleList()
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(fn=Attention(dim=dim,
                                     heads=heads,
                                     dim_head=dim_head,
                                     attn_dropout=attn_dropout,
                                     residual_dropout=residual_dropout), dim=dim),
                PreNorm(fn=FeedForward(dim=dim, 
                                       mult=ff_mult,
                                       dropout=ff_dropout), dim=dim)
            ]))
        
    def forward(self, x, return_attn=True, attention_mask=None):
        B,T = x.shape
        x_emb = self.embedding(x) # (B,T) -> (B,T, D)
        
        post_softmax_attentions = []
        for attn, ff in self.layers:
            attention_out, post_softmax_attn = attn(x_emb, attention_mask=attention_mask)
            post_softmax_attentions.append(post_softmax_attn)
            
            # residual connection
            x_emb = x_emb + attention_out # 
            
            ff_out = ff(x_emb)
            
            # residual connection
            x_emb  = x_emb  + ff_out 
            
        return x_emb, torch.stack(post_softmax_attentions)


# class TransformerDecoder
self_attn = pass
src_attn = pass

vocab_size = 10
dim = 4
heads = 6
dim_heads= 5
attn_dropout=0
residual_dropout = 0
x = torch.randint(0,vocab_size, size=(2,3))
emb = nn.Embedding(vocab_size,dim)
x_emb = emb(x) # (2, 3,4)

norm = PreNorm(dim=dim)

attn = Attention(dim=dim, 
                 heads=heads,
                 dim_head=dim_heads,
                 attn_dropout=0,
                 residual_dropout=0)
feedforward = FeedForward(dim=dim, mult=2)

class EncoderLayer(nn.Module):
    def __init__(self, 
                 dim,
                 heads,
                 dim_head,
                 attn_dropout=0,
                 residual_dropout=0,
                 ff_mult=2,
                 ff_dropout=0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(normalized_shape =dim)
        self.ff_norm = nn.LayerNorm(normalized_shape=dim)
        self.attn = Attention(dim, 
                              heads=heads,
                              dim_head=dim_head, 
                              attn_dropout=attn_dropout, 
                              residual_dropout=residual_dropout)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
    
    def forward(self, x, mask=None):

        # attention
        x_norm = self.attn_norm(x)
        attn_out = self.attn(query=x_norm,
                             key=x_norm,
                             value=x_norm,
                             attention_mask=mask)
        attn_out = x + attn_out
        # feed forward
        attn_out_norm = self.ff_norm(attn_out)
        ff_out = self.ff(attn_out_norm)
        ff_out = ff_out + attn_out
        return ff_out
    
el = EncoderLayer(dim=dim, 
             heads=heads,
             dim_head=dim_heads,
             attn_dropout=attn_dropout,
             residual_dropout=residual_dropout)
        
o = el(x_emb)      

class Encoder(nn.Module):
    def __init__(self, 
                 dim,
                 heads,
                 dim_head,
                 attn_dropout=0,
                 residual_dropout=0,
                 ff_mult=2,
                 ff_dropout=0,
                 n_layers=4):     
        super().__init__()
        self.layers = []
        for i in range(n_layers):
           self.layers.append(EncoderLayer(dim=dim,
                                           heads=heads,
                                           dim_head=dim_head,
                                           attn_dropout=attn_dropout,
                                           residual_dropout=residual_dropout,
                                           ff_mult=ff_mult,
                                           ff_dropout=ff_dropout))
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

encoder = Encoder(dim=dim,
                  heads=heads,
                  dim_head=dim_heads,
                  attn_dropout=attn_dropout,
                  residual_dropout=residual_dropout,
                  ff_mult=2,
                  ff_dropout=0,
                  n_layers=4)

pad = 10
src_mask = (x != pad).unsqueeze(-2).float()           
encoder_o = encoder(x_emb, mask=src_mask)
encoder_o.shape 
 
 
##### Decoder Layer ######
# Self attention -> cross attention -> feedforward
# attn query, key = memory from encoder
# value = attnout from deocder input
decoder_emb = nn.Embedding(vocab_size, dim)
x.shape == (2,3)
x_emb_decoder = decoder_emb(x)
self_attn_norm = nn.LayerNorm(normalized_shape=dim)

# Self Attention
self_attn = Attention(dim=dim,
                 heads=heads,
                 dim_head=dim_heads,
                 attn_dropout=attn_dropout,
                 residual_dropout=residual_dropout)
ff_norm = nn.LayerNorm(normalized_shape=dim)
ff = FeedForward(dim=dim, mult=2)

x_emb_decoder_norm = self_attn_norm(x_emb_decoder)
self_attn_decoder = self_attn(query=x_emb_decoder_norm, key=x_emb_decoder_norm, value=x_emb_decoder_norm)
# residual connection  
self_attn_decoder = self_attn_decoder + x_emb_decoder # (before norm)
# cross attention

cross_attn_memory_norm = nn.LayerNorm(normalized_shape=dim)
cross_attention_value_norm =  nn.LayerNorm(normalized_shape=dim)
cross_attn =  Attention(dim=dim,
                 heads=heads,
                 dim_head=dim_heads,
                 attn_dropout=attn_dropout,
                 residual_dropout=residual_dropout)
memory_norm = cross_attn_memory_norm(encoder_o)
decoder_value_norm = cross_attention_value_norm(self_attn_decoder)
causal_mask = torch.tril(torch.ones(3,3))

decoder_cross_attn = cross_attn(query=memory_norm,
                                key=memory_norm, 
                                value=decoder_value_norm,
                                attention_mask = causal_mask)


decoder_cross_attn = decoder_cross_attn + self_attn_decoder

# Feed Forward
decoder_cross_attn_norm = nn.LayerNorm(normalized_shape=dim)
ff = FeedForward(dim=dim, mult=2)

decoder_ca_norm = decoder_cross_attn_norm(decoder_cross_attn)
decoder_ff = ff(decoder_ca_norm)

decoder_ff = decoder_ff + decoder_cross_attn 



# attn_decoder_norm = ff_norm(attn_decoder)
# ff_decoder = ff(attn_decoder_norm)
# ff_decoder = ff_decoder + attn_decoder


 
# residual connection is always before the norm
            
o1 = norm(x_emb, fn=lambda x: attn(query=x, key=x, value=x))
o2  = norm(x=o1, fn= lambda x: feedforward(x))


# Encoder layer
# Norm(x) into transformer
fn = lambda x: Attention(dim=dim, heads=heads,
                        dim_head=dim_heads,
                        attn_dropout=0,
                        residual_dropout=0)(x, x,x)



ff = FeedForward(dim=5, mult=2, dropout=0)

out_x = PreNorm(dim=dim)(x_emb)
nn.Sequential(PreNorm(dim=5),
              Attention(dim=dim, heads=heads,
                        dim_head=dim_heads,
                        attn_dropout=0,
                        residual_dropout=0)
              
)
norm(x_emb, lambda x: fn(x, x, x)).shape




