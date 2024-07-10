import torch
import torch.nn as nn
import time

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0
mask = subsequent_mask(size=4)

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
    

def data_gen(V, batch_size, nbatches, timesteps=4):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, timesteps))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)    


timesteps = 4
sample = next(iter(data_gen(V=10, batch_size=2, nbatches=3, timesteps=timesteps)))
sample.src.shape # (B,T) (Encoder shifted target)
sample.tgt.shape # (B, T-1) (Decoder shifted target)
sample.tgt_y.shape # (B, T-1) (Decoder shifted target)
sample.tgt_mask.shape #(B, T, T)

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

    
    
def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


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
         self.debug = False
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
        if self.debug:
            print('Attention init.')
        q, k, v = self.query(query), self.key(key), self.value(value) # (B,T,D)  - > (B,T, inner_dim)
        q, k, v = (q.reshape(B, -1, self.heads, self.dim_head).transpose(1,2),\
                k.reshape(B, -1, self.heads, self.dim_head).transpose(1,2),\
                v.reshape(B, -1, self.heads, self.dim_head).transpose(1,2),) # (B, T, inner_dim) -> (B,T,H,dim_head) -> (B, H, T, dim_head)
        if self.debug:
            print('Before matmul q:{q.shape},k:{k.shape}') 
        sim = torch.matmul(q, k.transpose(-1,-2)) # (B, H, T, DH), (B, H, DH, T) -> (B, H, T ,T)
        sim = sim * self.scale
        # self.sim = sim 
        if attention_mask is not None:
            extended_attention_mask = invert_attention_mask(attention_mask)
            # self.extended_attention_mask = extended_attention_mask
            if self.debug:
                print(f'extended_attention_mask.shape: {extended_attention_mask.shape}')
                print(f'sim.shape: {sim.shape}')
            sim = sim + extended_attention_mask  # positions with attention_mask == 0 has very large negative value
        if self.debug:
            print('mask added to sim')
        sim = nn.functional.softmax(sim, dim=-1)
        attention = self.attn_dropout(sim)
        if self.debug:
            print(f'Attention.shape: {attention.shape}, v.shape: {v.shape}')
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

class PreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape = dim)
    def forward(self, x, fn):
        return fn(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout, residual_dropout, ff_mult, ff_dropout):
        super().__init__()
        self.pre_norms = nn.ModuleList([nn.LayerNorm(normalized_shape=dim) for _ in range(2)])
        self.attn = Attention(dim=dim, 
                              heads=heads,
                              dim_head=dim_head,
                              attn_dropout=attn_dropout,
                              residual_dropout=residual_dropout)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        
    def forward(self, x, src_mask=None):
        # Attn Out
        x_norm = self.pre_norms[0](x)
        x = self.attn(query=x_norm, key=x_norm, value=x_norm, attention_mask=src_mask) + x
        ## FF out
        x = self.ff(self.pre_norms[1](x)) + x
        return x
    

class Encoder(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout, residual_dropout, ff_mult, ff_dropout, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(EncoderLayer(dim=dim,
                                            heads=heads,
                                            dim_head=dim_head,
                                            attn_dropout=attn_dropout,
                                            residual_dropout=residual_dropout,
                                            ff_mult=ff_mult,
                                            ff_dropout=ff_dropout,
                                            ))
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask = src_mask)
        return x    
        
vocab_size = 10
batch_size = 2
block_size= 3
dim = 4
heads = 5
dim_head = 6
dropout = 0 
ff_mult=2
PAD_TOKEN = 2
x = torch.randint(0, vocab_size, size=(batch_size, block_size))
x = torch.tensor([[3,5,2],[1,3,8]])
emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim)
x_emb = emb(x)

e_layer = EncoderLayer(dim=dim,
                       heads=heads,
                       dim_head=dim_head,
                       attn_dropout=dropout,
                       residual_dropout=dropout,
                       ff_mult=ff_mult,
                       ff_dropout=dropout)

src_mask = (x != PAD_TOKEN).float()
e_out = e_layer(x_emb, src_mask=src_mask)

encoder = Encoder(dim=dim,
                       heads=heads,
                       dim_head=dim_head,
                       attn_dropout=dropout,
                       residual_dropout=dropout,
                       ff_mult=ff_mult,
                       ff_dropout=dropout,
                       n_layers=3)

encoder_o = encoder(x=x_emb, src_mask=src_mask)

# Decoder input
tgt_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim)

x
tgt = x[:,:-1] # (B, T-1)
tgt_y = x[:,1:]
tgt_mask = (tgt != PAD_TOKEN)
tgt_ts = tgt.shape[1]
tgt_causal_mask = torch.tril(torch.ones(tgt_ts, tgt_ts)) # (1,T-1, T-1) = (2,2)
tgt_mask = tgt_mask.unsqueeze(-2) & tgt_causal_mask.type_as(tgt_mask.data) # (B, T-1, 1), (T-1, T-1) -> (B, T-1, T-1)
assert tgt_mask.shape == (2,2,2)

src_mask.shape
src_mask.unsqueeze(-2).shape

tgt_emb = tgt_embedding(tgt)
assert tgt_emb.shape == (2,2,4) # (B, T-1, dim)

class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout, residual_dropout, ff_mult, ff_dropout):
        super().__init__()
        self.pre_norms = nn.ModuleList([nn.LayerNorm(normalized_shape=dim) for _ in range(3)])
        self.memory_norm = nn.LayerNorm(dim)
        self.self_attn = Attention(dim=dim, 
                              heads=heads,
                              dim_head=dim_head,
                              attn_dropout=attn_dropout,
                              residual_dropout=residual_dropout)
        # self.self_attn.debug = True        
        self.cross_attn =  Attention(dim=dim, 
                              heads=heads,
                              dim_head=dim_head,
                              attn_dropout=attn_dropout,
                              residual_dropout=residual_dropout)
        # self.cross_attn.debug = True        
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 1. self attention
        x_norm= self.pre_norms[0](x)
        # print(f'Before adding x:{x.shape}')
        x = self.self_attn(query=x_norm, key=x_norm, value=x_norm, attention_mask=tgt_mask)

        # print(f'point1: x{x.shape}')
        # 2. cross attention
        x_norm = self.pre_norms[1](x)
        memory = self.memory_norm(memory) 
        # print(f'mmeory:{memory.shape}, x_norm:{x_norm.shape}')
        x = self.cross_attn(query=x_norm, key=memory, value=memory, attention_mask=src_mask) 
        # print('point2')
        # 3. Feed Forard
        x_norm= self.pre_norms[2](x)
        x = self.ff(x_norm) + x
        # print('point3')
        return x

d_layer = DecoderLayer(dim=dim,
                       heads=heads,
                       dim_head=dim_head,
                       attn_dropout=dropout,
                       residual_dropout=dropout,
                       ff_mult=ff_mult,
                       ff_dropout=dropout)

src_mask = (x!=PAD_TOKEN).float()
src_mask.shape == (2,3) # (B, T)
tgt_mask.shape == (2,2,2) # (B, T-1, T-1)
encoder_o.shape == (2,3,4) # (B, T, D)
assert tgt_emb.shape == (2,2,4) # (B, T-1, D)

encoder_o.shape
# d_layer_o = d_layer(x=tgt_emb[:,0:1,:], memory=encoder_o, src_mask=src_mask.unsqueeze(-2), tgt_mask=tgt_mask.float()) 
d_layer_o = d_layer(x=tgt_emb[:,0:1,:], memory=encoder_o, src_mask=src_mask.unsqueeze(-2), tgt_mask=subsequent_mask(1).float()) 
d_layer_o.shape
d_layer_o
# assert d_layer_o.shape

class Decoder(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout, residual_dropout, ff_mult, ff_dropout, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(DecoderLayer(dim=dim, 
                                            heads=heads,
                                            dim_head=dim_head,
                                            attn_dropout=attn_dropout,
                                            residual_dropout=residual_dropout,
                                            ff_mult=ff_mult,
                                            ff_dropout=ff_dropout))
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x=x, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return x


decoder_n_layers = 3
decoder = Decoder(dim=dim,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=dropout,
                residual_dropout=dropout,
                ff_mult=ff_mult,
                ff_dropout=dropout,
                n_layers=decoder_n_layers)

decoder_o = decoder(x=tgt_emb, memory=encoder_o, src_mask=src_mask.unsqueeze(-2), tgt_mask=tgt_mask.float())

class Generator(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(dim, vocab_size)
   
    def forward(self,x):
        return F.log_softmax(self.linear(x), dim=-1)

generator =  Generator(dim=dim, vocab_size=vocab_size)
p = generator(decoder_o[:,-1,:]) 


class EncoderDecoder(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, dim, heads, dim_head, attn_dropout, residual_dropout, ff_mult, ff_dropout, n_encoder_layers, n_decoder_layers):
        super().__init__()  
        self.encoder = Encoder(dim=dim, 
                               heads=heads,
                               dim_head=dim_head,
                               attn_dropout=attn_dropout,
                               residual_dropout=residual_dropout,
                               ff_mult=ff_mult,
                               ff_dropout=ff_dropout,
                               n_layers=n_encoder_layers,
                               )
        self.decoder = Decoder(dim=dim, 
                               heads=heads,
                               dim_head=dim_head,
                               attn_dropout=attn_dropout,
                               residual_dropout=residual_dropout,
                               ff_mult=ff_mult,
                               ff_dropout=ff_dropout,
                               n_layers=n_decoder_layers,
                               )
        
        self.src_emb = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=dim)
        self.tgt_emb = nn.Embedding(num_embeddings=tgt_vocab_size, embedding_dim=dim)
        self.generator = Generator(dim=dim, vocab_size=tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(x=src, src_mask=src_mask)
        return self.decode(x=tgt, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
    
    def encode(self, x, src_mask):
        # return self.encoder(x=x, src_mask=src_mask)
        return self.encoder(x=self.src_emb(x), src_mask=src_mask)
    
    def decode(self, x, memory, src_mask, tgt_mask):
        # return self.decoder(x=x, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask) 
        return self.decoder(x=self.tgt_emb(x), memory=memory, src_mask=src_mask, tgt_mask=tgt_mask) 
        

src_mask = (x!=PAD_TOKEN).float()      
encoder_decoder = EncoderDecoder(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, dim=dim, heads=heads, dim_head=dim_head, 
                                 attn_dropout=dropout, residual_dropout=dropout, ff_mult=2, ff_dropout=dropout, 
                                 n_encoder_layers=3, n_decoder_layers=3)

assert x.shape == (2,3)
assert tgt.shape == (2,2)

# encoder_decoder_o = encoder_decoder(src=x_emb, tgt=tgt_emb, src_mask=src_mask.unsqueeze(-2), tgt_mask=tgt_mask.float()) 
tgt_mask.shape
src_mask.shape
encoder_decoder_o = encoder_decoder(src=x, tgt=tgt, src_mask=src_mask.unsqueeze(-2), tgt_mask=tgt_mask.float()) 
encoder_decoder_o.shape

vocab_size

src =  torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 1]])
src_mask = torch.ones(src.shape)
assert src.shape == (1,10)
src_mask.shape

dropout=0.1
encoder_decoder = EncoderDecoder(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, dim=512, heads=8, dim_head=512, 
                                 attn_dropout=dropout, residual_dropout=dropout, ff_mult=2, ff_dropout=dropout, 
                                 n_encoder_layers=2, n_decoder_layers=2)
# This was important from their code.
# Initialize parameters with Glorot / fan_avg.
for p in encoder_decoder.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
encoder_decoder.eval()
memory = encoder_decoder.encode(x=src, src_mask=src_mask)
# assert memory.shape == (1, 10, 4)
ys = torch.zeros(1,1).type_as(src)
# assert ys.shape == (1,1)

ys = torch.zeros(1,1).type_as(src)

for i in range(9):
    tgt_mask = subsequent_mask(ys.size(1)) # & (ys!=PAD_TOKEN).unsqueeze(-2)
    tgt_mask.shape
    out = encoder_decoder.decode(x=ys, memory=memory, src_mask=src_mask.float(),tgt_mask=tgt_mask.float())
    prob = encoder_decoder.generator(out[:,-1,:]) 
    # print(prob[:5])
    _, nextword = torch.max(prob, dim=-1)
    # print(nextword.data)
    ys = torch.cat([ys, nextword.clone().reshape(1,1).type_as(ys)], dim=1)
    print(f'index({i}): ys: {ys}') 




class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
    
    
class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed
    
def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask.float(), batch.tgt_mask.float()
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

        
        
def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


model = EncoderDecoder(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, dim=512, heads=8, dim_head=512, 
                                 attn_dropout=dropout, residual_dropout=dropout, ff_mult=2, ff_dropout=dropout, 
                                 n_encoder_layers=2, n_decoder_layers=2)
# This was important from their code.
# Initialize parameters with Glorot / fan_avg.
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

data_iter = data_gen(V=9, batch_size=20, nbatches=60)
next(iter(data_iter)).src
optimizer = torch.optim.Adam(model.parameters())

import numpy as np
# loss = nn.CrossEntropyLoss()
n_epoch = 1
for e in range(n_epoch):
    losses = []
    for i, batch in enumerate(data_iter):
        o = model(src=batch.src, tgt=batch.tgt, src_mask=batch.src_mask.float(), tgt_mask=batch.tgt_mask.float())
        out = model.generator(o)
        # out.shape
        # out.view(-1,10).shape
        # batch.tgt_y.shape
        loss = F.cross_entropy(out.reshape(-1,10), batch.tgt_y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f'epoch: {e}, avg_loss: {np.mean(losses)}')

        

        

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys

    

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
        
# Train the simple copy task.
from torch.optim.lr_scheduler import LambdaLR
def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None
            
def example_simple_model():
    V = 8
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = encoder_decoder

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_emb.embedding_dim, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))

example_simple_model()
# execute_example(example_simple_model)