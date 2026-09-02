from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm


from model import LlamaConfig, Transformer

class LLaMA:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    @staticmethod
    def build(checkpoints_dir, tokenizer_path, load_model,max_seq_len, max_batch_size, device):
        prev_time = time.time()
        if load_model():
            checkpoints  = sorted(Path(checkpoints_dir).glob('*.pth'))
            assert len(checkpoints) > 0, "No checkpoint files found"
            chk_path = checkpoints[0]
            print(f'Loading checkpoint {chk_path}')
            checkpoints = torch.load(chk_path, map_location = 'cpu')
            print(f'Loaded checkpoint in {(time.time() - prev_time):.2f}s')
            prev_time = time.time()
        with open(Path(checkpoints_dir) / 'params.json','r') as f:
            params = json.loads(f.read())
        config = LLamaConfig(
            max_seq_len = max_seq_len,
            max_batch_size = max_batch_size,
            device = device,
            dim = n_embd
            **params
        )
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        config.vocab_size = tokenizer.vocab_size()
        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        model = Transformer(config).to(device)
        if load_model:
            del checkpoint['rope.freqs']
            model.state_dict(checkpoint, strict = True)
            print(f'Loaded state dict in {time.time() - prev_time():.2f}s')
        return LlaMA(model, tokenizer, model_args)
if __name__ == '__main__':
    torch.manual_seed(0)
    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LlaMA.build(
        checkpoint_dir = 'llama-2-7b/',
        tokenizer_path = 'tokenizer.model',
        load_model = True,
        max_seq_len = 1024,
        max_batch_size = len(prompts),
        device = device,
    )

    prompts = [
        ''
    ]


    #Inference the Model



    print('All OK')

