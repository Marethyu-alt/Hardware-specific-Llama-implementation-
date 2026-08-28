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
    vocab_size: int = -1 #will change after loading tokenizer, likely 50k + 
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


if __name__ == '__main__': 
    print('This should theoretically crash the script via assertion error...')
    config = LlamaConfig()
    model = Transformer(config)