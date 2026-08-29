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
    vocab_size: int = -1 #will change after loading tokenizer, 30k
    multiple_of:int = 256 #multiplication constant
    ffn_dim_multiplier: Optional[float] = None #likely going to be 4 during MLP
    norm_eps:float = 1e-5 #random small epsilon norming
    max_batch_size:int = 32 #B = 32
    max_seq_len: int = 2048  #T = 2048
    device:str = None #move to CPU by default later, weights pre-loaded 

#main transformer
class Transformer(nn.Module): 
    def __init__(self,config): #takes as arg config: LlamaConfig will be passed into config later 
        super().__init__()
        assert config.vocab_size != -1, 'Vocab Size must be set'
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList()
        for _ in range(self.config.n_layers):
            self.layers.append(EncoderBlock(config)) #pass into Block to be created later for layering transformer
        self.norm = RMSNorm(config.n_embd, eps = config.norm_eps)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        self.freqs_complex = precompute_theta_pos_frequencies(self.config.n_embd // self.config.n_heads, self.config.max_seq_len * 2, device = self.config.device) #rope init
    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, 'KV cache requires seq len = 1' 
        h = self.tok_embeddings(tokens) #(B,T) * (C, n_embd) = (B,T,n_emmbd) dictionary lookup (32,2048,4096)
        freqs_complex = self.freqs_complex[start_pos: star_pos + seq_len] #ensuring dynamic accumulation of tokens also kv cache
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex) #consecutively apply the encoder to all layers
        h = self.norm(h)
        output = self.output(h).float()
        return output #transformer skeleton complete 
    




if __name__ == '__main__': 
    print('This should theoretically crash the script via assertion error...')
    config = LlamaConfig()
    model = Transformer(config)