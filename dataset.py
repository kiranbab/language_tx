import torch 
import torch.nn 
from torch.utils.data import Dataset 

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len) -> None:
        super().__init__() 
        self.ds = ds 
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        