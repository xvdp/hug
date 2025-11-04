"""

Toy transfomer 

Tokenizer:
    Tok                 AutoTokenizer.from_pretrained("gpt2")
    ToyTransformer      1 layer transfomer
    PositionalEncoding
    RoPE                RotationPositionEncoding
    RoPEScaling         in minitransformer.ipynb and ../docjumble/Attention.md   https://kaiokendev.github.io/til#extending-context-to-8k
    FFN                 Transformer Feed Forward Network
    Attention           Single head Attention Modulle
    MultiHeadAttention
    
    MiniTransformer     One block attention
    TokenizedDataset
    FlashAttention


"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

import matplotlib.pyplot as plt


text="""It stands. What? Yes. Say it stands. Had to up in the end and stand. Say bones. No bones but say bones.
Say ground. No ground but say ground. So as to say pain. No mind and pain? Say yes that the bones
may pain till no choice but stand. Somehow up and stand. Or better worse remains. Say remains of
mind where none to permit of pain. Pain of bones till no choice but up and stand. Somehow up.
Somehow stand. Remains of mind where none for the sake of pain. Here of bones. Other examples if
needs must. Of pain. Relief from. Change of.
"""

class Tok:
    """
    >>> T = Tok()
    >>> T.encode("this is a sentence") -> tensor([[1212,  318,  257, 6827,   13]])
    """
    def __init__(self, model="gpt2", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model)  # of the shelf tok vocab size 510257
        self.tokenizer.pad_token = self.tokenizer.eos_token                # simple padding strategy
        device = 'cpu' if not torch.cuda.is_available() or device is None else device
        self.device = torch.device(device)
        self.vocab_size = self.tokenizer.vocab_size,
    def encode(self, text, n_tokens_keep=64):
        return self.tokenizer(text, return_tensors="pt", truncation=True,
                max_length=n_tokens_keep).input_ids.to(self.device)
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)



class ToyTransformer:
    """
    one layer transfomer
    : Change d_model to 64 or 256 and watch the loss curve.
    : causal mask inside Attention for true autoregressive training.
    : Stack more blocks: self.blocks = nn.ModuleList([Block(d_model) for _ in range(n_layers)]).
    : sinusoids with learned positional embeddings (nn.Embedding(max_len, d_model)).

    from minitransformer import *
    T = ToyTransformer()
    T.step(text)
    """
    def __init__(self, d_model = 128, n_tokens_keep = 64, n_heads=1, device="cuda"):
        """ d_model     attention_is_all_you_need    512
                        BERT                         768
                        newer              1024,1280, 2048 ...
        
        
        """
        if not torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # of the shelf tok vocab size 510257
        self.tokenizer.pad_token = self.tokenizer.eos_token                # simple padding strategy

        self.model =  MiniTransformer(self.tokenizer.vocab_size, d_model).to(self.device)

        self.n_tokens_keep = n_tokens_keep
        self.lossmap = []

    def encode(self, text, n_tokens_keep):
        return self.tokenizer (text, return_tensors="pt", truncation=True,
                max_length=n_tokens_keep).input_ids.to(self.device)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def step(self, text, i=0):
        self.model.train()
        ids  = self.encode(text, self.n_tokens_keep+1)
        # print(ids.shape)          # (1, T)
        logits, loss = self.model(ids[:, i:i+self.n_tokens_keep], ids[:, i+1:i+1+self.n_tokens_keep]) # language-model targets = shifted input
        self.lossmap.append(loss.item())
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        loss.backward()
        optimizer.step()

        self.model.eval()
        with torch.no_grad():
            logits, loss = self.model(ids[:, i:i+self.n_tokens_keep], ids[:, i+1:i+1+self.n_tokens_keep]) # language-model targets = shifted inpu
            self.lossmap.append(loss.item())

    def infer(self,):
        self.model.eval()
        with torch.no_grad():
            prompt = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)  # start token
            max_new = 40
            with torch.no_grad():
                for _ in range(max_new):
                    logits, _ = self.model(prompt)
                    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    prompt  = torch.cat([prompt, next_id], dim=1)
        return self.decode(prompt)
            # print("\nGenerated:")
            # print(self.decode(prompt[0]))


def encode(text, tokenizer, n_tokens_keep=64, device="cpu"):
    return tokenizer (text, return_tensors="pt", truncation=True,
            max_length=n_tokens_keep).input_ids.to(device)

def decode(ids, tokenizer):
    return tokenizer.decode(ids, skip_special_tokens=True)

# ---- Sinusoidal positional encoding ----------
class PositionalEncoding(nn.Module):
    """ max_length of tokens, in this case if max_tokens are 64, thats all one needs
    .pe.shape = [1, max_len, model_d]
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)          # (max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)
        self.max_len = max_len
        self.d_model = d_model
    def forward(self, x):          # x: (batch, seq, d_model)
        return x + self.pe[:, :x.size(1)]
    def show(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.pe[0].T.numpy())
        plt.title("pos encoding: sin(pos,div),cos(pos,div)")
        plt.xlabel("position: to max tokens")
        plt.ylabel("d_model: div = log interp(1 to 1e-4)")
        plt.xticks([0,self.max_len])
        plt.yticks([0,self.d_model])
        plt.show()
    
import matplotlib.pyplot as plt

# --- Rotary position encoding
class RoPE(nn.Module):
    """
    2D RoPE (Su et al. 2021) for self-attention q, k.
    Call .rotate(q, k)  ->  q', k'  (same shapes)
    head_dim = d_model // n_heads # 512/8 -> 64
        GPT-NeoX, LLaMA-7B:  d_model = 4096, n_heads = 32 → head_dim = 128
        high end head_dim->256, 

    [x']   [cos iθ  -sin iθ][x]
    [y'] = [sin iθ   cos iθ][y]    (keeps norm, only rotates) 

    < R(i)q , R(j)k >  =  q.T R(i).T  R(j) k
                       =  q.T R(j-i) k
                       otation group property -> only j-i matters
    Unlike a precomputed sin cos table, rotations are continuous:

    Interpolation: one can train with 2000 tokens but interploate with 32,000
    Interploatinb base frequency (base (trainlen/target len)) rescales angular speed, 
    * CodeLLaMA, LLaMA-2-32 k, Mistral-7B-32 k all use this one-liner
    * 
    """
    def __init__(self, dim, base=10_000, device=None):
        super().__init__()
        # ![log interpolation between 1 and 1/base](./images/inv_freq.png)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq)          # (dim/2,)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)  # (seq_len,)
        freqs = torch.outer(t, self.inv_freq)                                # (seq_len, dim/2)
        return torch.polar(torch.ones_like(freqs), freqs)                    # (seq_len, dim/2) complex

    def rotate(self, q, k):
        """
        q, k:  (batch, n_heads, seq_len, head_dim)
        returns rotated q, k (same sh 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))ape)
        """
        seq_len, head_dim = q.shape[-2], q.shape[-1]
        assert head_dim % 2 == 0
        # build rotation matrix once per forward
        rot_mat = self.forward(seq_len, q.device)                            # (seq_len, head_dim/2)
        # reshape last dim into pairs
        q_  = q.float().reshape(*q.shape[:-1], -1, 2)                        # (..., head_dim/2, 2)
        k_  = k.float().reshape(*k.shape[:-1], -1, 2)
        # complex multiply  (a+bi) * e^(iθ)  ==  real view
        q_complex = torch.view_as_complex(q_)                                # (..., head_dim/2)
        k_complex = torch.view_as_complex(k_)
        q_out = torch.view_as_real(q_complex * rot_mat).flatten(-2)          # (..., head_dim)
        k_out = torch.view_as_real(k_complex * rot_mat).flatten(-2)
        return q_out.type_as(q), k_out.type_as(k)


class RoPEScaling(torch.nn.Module):
    """ https://kaiokendev.github.io/til#extending-context-to-8k
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        max_position_embeddings = 8192
        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        # These two lines:
        self.scale = 1 / 4
        t *= self.scale


def _exp(angle):
    """e^(i·angle)  (angle real, radians) -> complex tensor"""
    return _rexp(torch.ones_like(angle), angle)

def _rexp(real, angle):
    """e^(i·angle)  (angle real, radians) -> complex tensor"""
    return torch.polar(real, angle)

class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_up = nn.Linear(d_model*4, d_model, bias=True)  
        self.w_dn = nn.Linear(d_model, 4*d_model, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.dn(self.relu(self.w_up(x)))


# ---- Single-head scaled dot-product attention ----------
class Attention(nn.Module):
    """
    W_Q = np.random.randn(d_model, d_k) 
    W_K = np.random.randn(d_model, d_k)
    W_V = np.random.randn(d_model, d_v)

    1. Tokenize input sentences
    Input Text -> Token List[int]:                  shapoe (seq_len) sequence or context length.  512 in original Transformer, 2048 in LLAMA, ..., extensible to 1M
    Input Text Batch -> batch of sentences          shape (B, seq_len)

    2. Build input from tokenized indexed embedding
    Model Parameters
        Embedding: learns token vocabulary      shape (vocab_size, model_dim), vocab_size: 50K to 500K tokens, extensible, model_dim: 2048
    
    x = Embedding[Token_List Batch] ->      shape (B, seq_len model_dim), token indexed vectors from Embedding  Emb_w

    3. Attention mechanism: Reorder information and identify what to attend to
        xa = LayerNorm(x) # with prenorm

        Parameters
            Q_w, K_w Query, Key, weights:   shape (model_dim, key_dim)      in single head attn (model_dim, model_dim), with MultiHead typically (model_dim, model_dim//num_heads)
            V_w Value weights:              shape (model_dim, value_dim)    similar to Query weights
            O_w Output weights:             shape (model_dim, model_dim) 
        
        build Query, Key, Values
            Q = xa @ Q_w     shape (B, seq_len, model_dim) @ (model_dim, key_dim) -> (B, seq_len, key_dim) 
            K = xa @ K_w     shape (B, seq_len, model_dim) @ (model_dim, key_dim) -> (B, seq_len, key_dim) 
            QKT = (Q @ K.T) / sqrt(key_dim)     shape (B, seq_len, key_dim) @ (B, key_dim, seq_len)  -> (B, seq_len, seq_len) 
                QKT is inner product: highlights what, given the products of Embedding, Query and Key Weights, is interesting
            sQKT = SoftmMax(QKT)   QKT is scaled down so as to not saturate weights, then softmaxed to convert score to probability.
            
            V = xa @ V_w     shape (B, seq_len, model_dim) @ (model_dim, value_dim) -> (B, seq_len, value_dim) 
            Attn = sQKT @ V shape (B, seq_len, seq_len) @ (B, seq_len, value_dim) -> (B, seq_len, value_dim)
            attn_out = Concat(Attn_head_0, ..., .Attn_head_n) @ O_w     shape (B, seq_len, model_dim)

    4. Add unnormed Residual
    x = attn_out + x     shape (B, seq_len, model_dim)

    5. FFN
        xf = LayerNorm(x)   shape (B, seq_len, model_dim)

        Parameters
            W_up    shape(model_dim*4, model_dim)
            W_dn    shape(model_dim, model_dim*4)

        xf = xf @ W_up      shape (B, seq_len, model_dim) @ (model_dim*4, model_dim) -> (B, seq_len, 4*model_dim)
        xf = max(0, xf)         non linear threshold
        ffn_out = xf @ W_dn     shape (B, seq_len, 4*model_dim) @ (model_dim, 4*model_dim) -> (B, seq_len, model_dim) 

    6. Add Residual from before FFN
    x = ffn_out + x     shape (B, seq_len, model_dim)

    7. New Attention block
    ...
    Last: 
        Parameters 
            Linear Weight = Embedding Weight, shared
            shape (vocab_size, model_dim),
    logits = x @ L_w    (B, seq_len, model_dim) @ (vocab_size, model_dim) - >  (B, seq_len,vocab_size)

    target: sequence shifted by 1 : (B, seq_len)
    XEntropy (B*seq_len, vocab_size) (B*seq_len) -> Loss

            
    """
    def __init__(self, d_model, verbose=False):
        super().__init__()

        d_k=d_v=d_model
        self.w_q = nn.Linear(d_model, d_k, bias=False)      # .weight (d_model, d_k)
        self.w_k = nn.Linear(d_model, d_k, bias=False)      # .weight (d_model, d_k)
        self.w_v = nn.Linear(d_model, d_v, bias=False)      # .weight (d_model, d_v)
        # MultiHead(Q, K, V ) = Concat(head_1, ..., head_h) Wo
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # .weight (d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(self, x, mask=None):
        #B, T, C = x.shape       # (B, num_tokens, d_model)
        # pos encoded, token embeddings, * weights.
        Q = self.w_q(x)             # (B, num_tokens, d_model) @ (d_model, d_model) -> (B, num_tokens, d_model) 
        K = self.w_k(x)
        V = self.w_v(x)
        # (B, num_tokens, d_k) @  (B, d_k, num_tokens) ->  (B, num_tokens, num_tokens)
        Q_KT = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            Q_KT = Q_KT.masked_fill(mask == 0, -1e9)
        attn = Q_KT.softmax(dim=-1)
        out = attn @ V # (B, num_tokens, num_tokens) @  (B, num_tokens, d_v) ->
        if self.verbose:
            print(f"  x {tuple(x.shape)} @ W_q {tuple(self.w_q.weight.shape)} -> Q {tuple(Q.shape)}")
            print(f"  x {tuple(x.shape)} @ W_k {tuple(self.w_k.weight.shape)} -> K {tuple(K.shape)}")
            print(f"  x {tuple(x.shape)} @ W_v {tuple(self.w_v.weight.shape)} -> V {tuple(V.shape)}")
            print(f"  Q {tuple(Q.shape)} @ K.T {tuple(K.transpose(-2, -1).shape)} -> Q.KT {tuple(Q_KT.shape)}  ")
            print(f"  softmax(Q.KT) {tuple(attn.shape)} @ V {tuple(V.shape)} ->  out {tuple(out.shape)}  ")
            print(f"  out {tuple(out.shape)} @ W_o {tuple(self.w_o.weight.shape)} ->  ")
        out = self.w_o(out)        # (B, num_tokens, d_model) @  (d_model d_model)
        return out


class MultiHeadAttention(nn.Module):
    """ EXPAND AND CONTRACT """
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads          # 64 for 512/8
        self.d_v = d_model // n_heads

        # single big matrix for each of Q,K,V  (bias is optional)
        self.w_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * self.d_v, bias=False)

        # output projection
        self.w_o = nn.Linear(n_heads * self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        B, T, _ = x.shape                       # batch, seq, d_model

        # 1. linear in one go
        Q = self.w_q(x)                         # (B,T,n_heads*d_k)
        K = self.w_k(x)
        V = self.w_v(x)

        # 2. reshape → separate heads
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B,n_heads,T,d_k)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_v).transpose(1, 2)

        # 3. scaled dot-product on the last two dims
        scores = (Q @ K.transpose(-2, -1)) * self.scale           # (B,n_heads,T,T)
        if mask is not None:                                      # causal or padding
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ V                                            # (B,n_heads,T,d_v)

        # 4. merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_v)

        # 5. final linear
        return self.w_o(out)                                      # (B,T,d_model)


# --------- One-block transformer ----------
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads=1, pre_norm=True, verbose=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # .num_embeddings -> vocab_size, .embedding_dim -> dmodel, .weight
        self.pos_enc = PositionalEncoding(d_model) # adds pos_enc.pe
        
        self.verbose = verbose
        #d_k = d_v = d_model//n_heads
        self.attn  = Attention(d_model, verbose=verbose)

        # "Attention is all you need" uses POST norm
        self.pre_norm = pre_norm   # prenorm trains longer more stably
        self.ln    = nn.LayerNorm(d_model)

        # feed forward         
        self.ffn = FFN(d_model)

        # output head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight sharing (tie input/output embeddings)

        self.init_weights()

    def init_weights(self):
        # GPT-2 / Xavier style
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: (B, n_tokens)
        #### Transformmer Block: Attention, FFN
        if self.verbose:
            print(f"transformer({idx.shape} {targets if targets is None else targets.shape})")
        x = self.embed(idx) # x = embedding.weight[idx]                     (B, n_tokens, d_model)
        x = self.pos_enc(x) # x = embedding[idx] + pencoding[:n_tokens]     (B, n_tokens, d_model)
        if self.verbose:
            print(f"x = embed + pos_enc :{x.shape}")
        if self.pre_norm:   
            x = x + self.attn(self.ln(x))   # x + attention(NORM (x encoding + embedding[idx])), where
            x = x + self.ffn(self.ln(x))
        else:              
            x = self.ln(x + self.attn(x))    # NORM(residual + attention(encoding + embedding[idx]))
            x = self.ln(x + self.ffn(x))
        
        # OUTPUT
        logits = self.lm_head(x)           # (B,T,vocab_size)
        if self.verbose:
            print(f"x + attn(ln(x)) {x.shape}")
            print(x.shape)
            print(logits.shape)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss


class TokenizedDataset(torch.utils.data.Dataset):
    """
    
batch_x, batch_y = next(iter(loader))   # both (batch, block_size)
logits, loss = model(batch_x, batch_y)
    """
    def __init__(self, token_ids, block_size):
        self.block_size = block_size
        self.tokens = token_ids                # long 1-D tensor

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx+self.block_size+1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def flash_attn(q, k, v, dropout_p=0.0, causal=False):
    """
    q, k, v: (batch, n_heads, seq_len, head_dim)
    Returns same shape, O(T) memory, uses Flash when HW+PT≥2.0
    """
    # Let PyTorch pick the fastest kernel (Flash, Mem-efficient, or C++)
    return F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=dropout_p if q.training else 0.0,
        is_causal=causal
    )


class FlashAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, causal=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.rope = RoPE(self.head_dim)
        self.dropout_p = dropout
        self.causal = causal

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)                     # 3×(B,T,d_model)
        q, k, v = map(lambda t: t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2), qkv)
        q, k = self.rope.rotate(q, k)                          # apply RoPE
        out = flash_attn(q, k, v, self.dropout_p, self.causal) # Flash or fallback
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
    

######################
#
# visualizations - moved to minitransformer.ipynb
#
def plot_freq(d_model, tohens):
    """plots position embedding coefficients for token """
    pos = torch.arange(0, tohens).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    # plt.figure(figsize=(8,4))
    # plt.subplot(121)
    plt.plot(div, label="$\\theta$s for token[0]", markevery=[0, d_model//2-1], marker='o');
    plt.plot(div*tohens, label="$\\theta$s for token[511]",  markevery=[0, d_model//2-1], marker='o');
    plt.yscale('log');plt.grid();plt.legend(title=f"e.g. seq len = {tohens}");plt.xticks([0, d_model//2-1], [0, f"{d_model//2-1}\n(d_model/2)"]);plt.xlabel("half Embedding depth");
    # plt.subplot(122)
    # plt.plot(div, label="$\\theta$s for token[0]", markevery=[0, 127], marker='o');plt.plot(div*512, label="$\\theta$s for token[511]",  markevery=[0, 127], marker='o');plt.yscale('log');plt.grid();plt.legend(title="e.g. seq len = 512");plt.xticks([0, 127], [0, "127\n(d_model/2)"]);plt.xlabel("half Embedding depth");
    plt.tight_layout();plt.show()

#plt.plot(inv_freq, markevery=[0, dim//2 -1], marker='o');plt.yscale('log');plt.xticks([0, dim//2-1]);plt.grid();plt.xlabel('half head dim');plt.title('log freq');plt.tight_layout();plt.show()


def plot_rot_ntk(d_model=128, max_emb=2048, to_max=16384, base=10_000):
    dim = d_model # head dimension
    div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    # min _div = 0.0001
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    # from https://colab.research.google.com/drive/1VI2nhlyKvd5cw4-zHvAIk00cAVj2lCCC why NTK scale the rotation
    a = to_max/max_emb
    basentk =base *a **(dim/dim-2)

    inv_freqntk = 1.0 / (basentk ** (torch.arange(0, dim, 2).float() / dim))
    plt.plot(inv_freq, markevery=[0, dim//2 -1], marker='o', label='token[0]');plt.xticks([0, dim//2-1]);
    plt.text(dim//2-4,inv_freq[-1], s=f'({dim//2-20}, {inv_freq[-1]:.4f})')
    plt.text(dim//2-4,inv_freqntk[-1], s=f'({dim//2-20}, {inv_freqntk[-1]:.5f})')
    plt.plot(inv_freq*2048, markevery=[0, dim//2 -1], marker='o', label='token[2048]')
    plt.plot(inv_freqntk, markevery=[0, dim//2 -1], marker='o', label='token[0]_scaled', linestyle="--")
    plt.plot(inv_freqntk*16384, markevery=[0, dim//2 -1], marker='o', label='token[16384]_scaled', linestyle="--")
    plt.legend()
    plt.grid();plt.xlabel('half head dim');plt.title('log freq');plt.yscale('log');plt.tight_layout();plt.show()


def plot_ntk_scaling_plane(dim=128, emb=2048, scale=8, base=10_000, show=True, **kwargs):
    """
    frequency plane
    NTK-aware scaling keeps the eigen-spectrum of the neural tangent kernel (the Gram matrix of gradients) unchanged 
    rotation-based feature φ(θ) = e^{i θ} the NTK at position t is built from
        ∂f/∂θ · ∂f/∂θ' ∝ cos(θ_t - θ_{t'}).
    Training  lenght L the distribution of angular differences Δ = |t - t'| spans [0, L].
    Inference:  L = λL (λ = 8) same distribution of phase difference: invariant kernel eigenvalues
    ntk compresses the lowest frequencies ( last embedding pos)
    """
    plt.title(f"Encoding freq plane - {scale}X stretch w NTK scale ")
    idcs = torch.arange(0,dim, 2)
    iff = 1.0 / (base ** (idcs.float() / dim))
    plt.plot(iff, linestyle="-", color="black", linewidth=2, label=f"trained context {emb:,}")
    plt.plot(iff*emb, linestyle="-",color="black", linewidth=1)#, label=f"trained at {emb}, tokens{dim-2, dim-1}")
    plt.plot([0,0], [1,emb], color="black", linewidth=0.75)
    plt.plot([dim//2-1, dim//2-1], [iff[-1],iff[-1]*emb], color="black", linewidth=0.75)
    plt.scatter(dim//2 -1, iff[-1].item(), label=f"base {base:,}: 1 to {iff[-1]:.4f}" )
    # NTK scalking to keep eigenvalues similar
    basentk = base* 8 **(dim/(dim-2))
    iff8 = 1.0 / (basentk ** (idcs.float() / dim))
    plt.plot(iff8, color="red", linestyle="--",  linewidth=2,  label=f"test context {emb*scale:,} (NTK_Scaled)")
    plt.plot(iff8*emb*scale, color='red', linestyle="--", linewidth=1)# label=f"NTK scaled, test at {emb*scale}, tokens{dim-2, dim-1}")
    plt.plot([0,0], [iff8[0], iff8[0]*emb*scale], color="red", linestyle="--", linewidth=0.75)
    plt.plot([dim//2-1, dim//2-1], [iff8[-1], iff8[-1]*emb*scale], color="red", linestyle="--", linewidth=0.75)
    plt.scatter(dim//2 -1, iff8[-1].item(), label=f"base {int(basentk):,}: 1 to {iff8[-1]:.7f}" )
    texts=kwargs.get("texts", [f"{2048*8:,}th test token", f"{2048:,}th train token", "1st test token", "1st train token"])
    poss=kwargs.get("poss", [(43, 0.6), (43, 0.11), (43, 1.8e-5), (43, 5e-4)])
    rots=kwargs.get("rots", [-22, -17, -22, -17])
    colorss=kwargs.get("colorss", ['red', 'black', 'red', 'black'])
    assert len(texts) == len(poss) == len(rots), f"{texts}, {poss}, {rots} must be of equal len"
    # plt.text(47, 0, s=f"{2048*8:,}th token", rotation=-22, color="red")
    # plt.text(47,-2, s=f"{2048:,}th token", rotation=-18, color="black")
    for i, txt in enumerate(texts):
        kw = {}
        if len(colorss) > i:
            kw['color'] = colorss[i]
        plt.text(poss[i][0], poss[i][1], s=txt, rotation=rots[i], **kw)
    plt.legend()
    plt.xticks([0, dim//2 - 1])
    plt.grid();plt.yscale('log')
    if show:
        plt.show()
        
    # plot_ntk_scaling_plane(texts=["asdfasdfasdfasdf"], poss=[(20, 12)], rots=[-22])
    # plot_ntk_scaling_plane(texts=[f"{2048*8:,}th token"], poss=[(26, 26)], rots=[-22], colorss=['red'])

    """
    >>> plot_ntk_scaling_plane(texts=[f"{2048*8:,}th token"], poss=[(45, 1)], rots=[-22], colorss=['red'])
>>> plot_ntk_scaling_plane(texts=[f"{2048*8:,}th token", f"{2048:,}th token"], poss=[(45, 5), (45, 0.1)], rots=[-22, -17], colorss=['red', 'black'])
"""


def compare_RoPE_Scaling(seq_len=2048, head_dim=128, base=10_000, scale=8, norm=True):
    lam_train = rope_ntk_spectrum(seq_len, head_dim, base, 1.).cpu()
    lam_test = rope_ntk_spectrum(seq_len*scale, head_dim, base, scale).cpu()
    lam_notnk = rope_ntk_spectrum(seq_len*scale, head_dim, base, 1.).cpu()
    lam_train = lam_train if not norm else lam_train / lam_train.sum()
    lam_test = lam_test if not norm else lam_test / lam_test.sum()
    lam_notnk = lam_notnk if not norm else lam_notnk / lam_notnk.sum()
    plt.title("Token Egenvalues of ")
    plt.plot(lam_train[:10], label=f"train freq eigvals at {seq_len:,} tokens")
    plt.plot(lam_test[:10], linestyle="--", color="black", label=f"ntk scaled freq eigvals at {seq_len*8:,} tokens")
    plt.plot(lam_notnk[:10], linestyle="--", color="red", label=f" eigvals at {seq_len*8:,} tokens")
    plt.legend()
    plt.grid()
    plt.show()


### Why NTK Scaling makes sense, maintains eighe value of the gram matrix
    """ 
    Gram matrix expresses the geometric relation between a set of vectors
        for rotations, it  means the relative rotations
        for unit vector z = e^iθ, z* = e^-iθ, so  (e^iθ)(e^-iθ) = e^(iθ-iθ) = e^0 = 1
    for Z = [e^iθ1,e^iθ2, ..., e^iθn]
    G = ZZ* G_{ij} = e^{i(θi-θj)} : each component is the difference betwen 2 rotations

    Eigen values of a matrix : det(Mat - λI) = 0
        ∣λ∣=1, arg(λ)=±θ
    The Eigen values of the Gram  of a set of rotations encodes the phase correlations
    largest eigenvalue corresponds to the direction (or row-combination) with the highest average phase correlation.
    G = VΛV* V Eigenvectors Λ: Eigenvalues - spatial Fourier or coherence analysis
    G_{mn} = ZZ* = ∑_k z_{mk} z*_{nk}
    Z = e^{iθ} = cos(θ) + i sin(θ)
    Z* = e^{iθ} = cos(θ) - i sin(θ)
    G_{mn} = e^{i(θ_m-θ_n)} = cos(θ_m-θ_n) + i sin(θ_m-θ_n)

    Z @ Z*.T = (cos(θ_m) + i sin(θ_m))*(cos(θ_n) - i sin(θ_n)) 
             = (cos(θ_m)*cos(θ_n)) + (sin(θ_m)*sin(θ_n)) + i(sin(θ_m)(cos(θ_n) - sin(θ_n)(cos(θ_m))
        R = cos(θ_m)*cos(θ_n) + sin(θ_m)*sin(θ_n) = cos(θ_m - θ_n)
        I = sin(θ_m)(cos(θ_n) - sin(θ_n)(cos(θ_m) = sin(θ_m - θ_n)
            e^{i(a-b)} = e^{ia} e^{-ib}
    cos(a-b) + i sin(a-b) = (cos(a) + i sin(a)) (cos(b) - i sin(b))
                          = cos(a)cos(b) + sin(a)sin(b) + i (sin(a)cos(b) - sin(b) cos(a))
        cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
        sin(a-b) = sin(a)cos(b) - sin(b)cos(a)
    since z_{mk} =  1 +isin(θ_{mk})
    G_{mn} = ∑_k[1 + isin(θ_{mk})][1 - i sin(θ_{nk}]
           = ∑_k[1 + sin(θ_{mk} sin(θ_{nk}) + i(sin(θ_{nnk}-i(sin(θ_{nk}))]
    G.real = ∑_k[1 + sin(θ_{mk} * sin(θ_{nk})]
    G.imag = ∑_k[i(sin(θ_{nnk} - i(sin(θ_{nk}))]
    """
def rope_ntk_spectrum(seq_len=2048, head_dim=128, base=10_000, rescale=1.0, device="cuda", drop_imag=True):
    """
    Returns eigen-values of the NTK Gram matrix for RoPE features.
    rescale = 1.0  → original base
              8.0  → NTK-aware stretch factor
        # quick demo
    L = 2048
    lam_orig = rope_ntk_spectrum(L, rescale=1.0)
    lam_stretch = rope_ntk_spectrum(L, rescale=8.0)
    print("Top-10 eigen-values:")
    print("original :", lam_orig[:10].tolist())
    print("NTK 8  :", lam_stretch[:10].tolist())
    """
    base = base * (rescale ** (head_dim / (head_dim - 2)))
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float, device=device)
    # phase matrix  (seq_len, head_dim//2)
    phases = t.outer(inv_freq)                     # ω_k * t
    # complex features  e^{i ω_k t}
    z = torch.polar(torch.ones_like(phases), phases)        # (T, d/2)
    # analytic NTK Gram matrix  G_{i,j} = Re[ <z_i, z_j> ]  (real part)
    G = z @ z.conj().T
    # eigen-decompose
    _G = G.real if drop_imag else G
    lam, _ = torch.linalg.eigh(_G)
    return lam.sort(descending=True)[0]



def plot_frequencies(dim=128, base=10_000, max_emb=2048, context_scale=8, device='cpu'):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_emb, device=device)
    freqs=torch.outer(t, inv_freq)
    # scale by 8 
    basentk =base *context_scale **(dim/dim-2)
    inv_freqntk = 1.0 / (basentk ** (torch.arange(0, dim, 2).float() / dim))
    tnk = torch.arange(max_emb*context_scale, device=device)
    freqsnk=torch.outer(tnk, inv_freqntk)
    f = freqs
    fnk =freqsnk

    plt.plot(f[:, 0], label="freqs, tokens 0, 1");plt.plot(f[:,dim//4-1], label=f"freqs tokens {dim//2-2, dim//2-1}");plt.plot(f[:,-1], label=f"freqs tokens {dim-2, dim-1}");
    # plt.grid();plt.yscale('log');plt.title(f"freq plane dim={dim}, max_tokens={max_emb*context_scale}");plt.legend();plt.show()
    plt.plot(fnk[:, 0], linestyle="--",label="freqs, tokens 0, 1");plt.plot(fnk[:,31],linestyle="--", label="freqs tokens 62,63");plt.plot(fnk[:,-1],linestyle="--", label="freqs tokens 126, 127");
    
    plt.grid();plt.yscale('log');plt.title("freq plane dim=128, max_tokens=2048*8");plt.legend();plt.show()
    return freqs, freqsnk

