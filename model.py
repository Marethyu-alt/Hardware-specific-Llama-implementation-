#Llama Architecture w/weights pre-loaded. implementation specific to NVIDIA 4080 using MUON 
#Default Setup, will change later 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class LlamaConfig:
    n_embd: int = 4096 #same as dim lmao **  #C = 4096
    n_layers:int = 32
    n_heads:int = 32
    n_kv_heads: Optional[int] = None # number of heads for the Keys and Queries
    vocab_size: int = 30_000 #will change after loading tokenizer, 30k
    multiple_of:int = 256 #multiplication constant
    ffn_dim_multiplier: Optional[float] = None #likely going to be 4 during MLP
    norm_eps:float = 1e-5 #random small epsilon norming
    max_batch_size:int = 32 #B = 32
    max_seq_len: int = 2048  #T = 2048
    device:str = None #move to CPU by default later, weights pre-loaded 


#Understanding ROPE: 
#Absolute Positional Encodings = xiWQ * xjWK.T / sqrt d z 
#Relative Positional Encodings (NOT USED IN LLAMA) = xiWQ(xjWk + aKij)T / sqrt dz  = xiWQ* xjWk.T/sqrt dz (content to content similarity) + xiWq*aKij.T / sqrt dz (content to relative position term)
#Fundamental Question: can we find an inner product or similarity over vectors of q and k such that the attention mechanism depends only on the two vectors and their distance?
#Rotary Position Embeddings: Attention Score = (q1k1 + q2k2)cos(m-n)theta + (q1k2-q2k1)sin(m-n)theta for a 2-d vector this results in a number that represents both meaning(semantic positions which is found through backprop and multi-head attention) and distance (position)
#Simplified Formula using complex numbers for computational efficiency: Attention Score = (q*k)*[e^i(m-n)theta] (Euler's formula)

#ROPE implementation
def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta: float = 10_000.0):
    assert head_dim % 2 == 0, 'Head dims must be even for ROPE'
    theta_numerator = torch.arange(0,head_dim,2).float() #formula theta = 10k ^  (-2(i-1) / head_dim) for i = 1,2,3..head_dim/2 = 1/10,000 ^ 2((i-1)/head_dim), setting up different speeds of theta
    theta = 1 / (theta ** (theta_numerator/ head_dim)).to(device) #(head_dim / 2) = shape
    m = torch.arange(seq_len, device = device)
    freqs = torch.outer(m, theta).float() #(seq_len, headdim/2), each individual element of m times all values of theta, all positions of tokens and their corresponding angles/frequencies
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex #matrix of (seq_len, head_dim/2) literally just positions of tokens and by how much each is rotated respectively
def apply_rotary_embeddings(x,freqs_complex, device):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2)) #(B, seq_len, H, head_dim) -> (B,seq_len,, head_dim/2 (real_dim, imag_dim)) by dividing by 2 then converting to imaginary, token features (x)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2) #adding 2 extra dims for B, H (1, Seq_len, 1, head_dim/2)
    x_rotated = x_complex * freqs_complex #multiplies actual tokens by their corresponding frequencies
    x_out = torch.view_as_real(x_rotated) #returning back to real numbers to be computed via gpu with ease
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

#Understanding RMS Norm:
#In layer norm we took standard gaussian, i.e. a normal distribution with a mean of 0 and var of 1 of the features (n_embd)
#In RMS Norm we only make the n_embd have a var of 1 (not gaussian!), formula: x_norm = x / sqrt (1/n * summation of xi^2 + epsilon) to get var (avg distance from mean) = 1
#rmsnorm class
class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
    def _norm(self,x): #where x is (B,T,C (n_embd)), formula: x_norm = x / sqrt (1/n * summation of xi^2 + epsilon)
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)

        

#main transformer
class Transformer(nn.Module): 
    def __init__(self,config): #takes as arg config: LlamaConfig will be passed into config later 
        super().__init__()
        self.config = config
        assert config.vocab_size != -1, 'Vocab Size must be set'
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd); print(self.tok_embeddings.weight.shape)
        self.layers = nn.ModuleList()
        for _ in range(self.config.n_layers):
           self.layers.append(EncoderBlock(config)) #pass into Block to be created later for layering transformer
        self.norm = RMSNorm(config.n_embd, eps = config.norm_eps)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        self.freqs_complex = precompute_theta_pos_frequencies(self.config.n_embd // self.config.n_heads, self.config.max_seq_len * 2, device = self.config.device) #rope init
    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape; print(tokens.shape)
        assert seq_len == 1, 'KV cache requires seq len = 1' 
        h = self.tok_embeddings(tokens) #(B,vocab_size) * (T, n_embd) = (B,T,n_emmbd) dictionary lookup (32,2048,4096)
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len] #ensuring dynamic accumulation of tokens also kv cache
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex) #consecutively apply the encoder to all layers
        h = self.norm(h)
        output = self.output(h).float() #RMS Norm (32,2048,4096) * (4096,30,000) = (32,2048,30,000) or (B,T,C)
        return output #transformer skeleton complete 
    


#test cases

if __name__ == '__main__': 
 #  print('This should theoretically crash the script via assertion error...')
    vocab_size = 30_000
    batch_size = 32
    seq_len = 512
    n_embd = 256
    dummy_tokens = torch.randint(0,vocab_size,(batch_size, seq_len))
    start_pos = 0
    config = LlamaConfig()
    model = Transformer(config)
    output = model(tokens = dummy_tokens, start_pos = start_pos)
    print(output.shape)
    