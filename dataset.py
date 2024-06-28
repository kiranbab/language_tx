from typing import Any
import torch 
import torch.nn 
from torch.utils.data import Dataset 

class BilingualDataset(Dataset):
    """
    A dataset class for handling bilingual data, specifically designed for sequence-to-sequence models.
    
    Attributes:
        ds (Dataset): The source dataset containing bilingual sentence pairs.
        tokenizer_src (Tokenizer): Tokenizer for the source language.
        tokenizer_tgt (Tokenizer): Tokenizer for the target language.
        src_lang (str): Source language code.
        tgt_lang (str): Target language code.
        seq_len (int): The fixed sequence length for both source and target sentences.
        sos_token (torch.Tensor): Start-of-sentence token for both source and target languages.
        eos_token (torch.Tensor): End-of-sentence token for both source and target languages.
        pad_token (torch.Tensor): Padding token for both source and target languages.
    
    Methods:
        __len__() -> int:
            Returns the number of items in the dataset.
        
        __getitem__(index) -> Any:
            Returns a dictionary containing processed source and target sentences, along with their masks and labels.
            Raises ValueError if the sentence length exceeds `seq_len`.
    
    Parameters:
        ds (Dataset): The dataset containing bilingual sentence pairs.
        tokenizer_src (Tokenizer): The tokenizer for the source language.
        tokenizer_tgt (Tokenizer): The tokenizer for the target language.
        src_lang (str): The source language code.
        tgt_lang (str): The target language code.
        seq_len (int): The fixed sequence length for processing.
    
    Returns:
        A dictionary containing the following keys:
            - "encoder_input": The tokenized and padded source sentence.
            - "decoder_input": The tokenized and padded target sentence for the decoder input.
            - "encoder_mask": The attention mask for the encoder input.
            - "decoder_mask": The attention mask for the decoder input, including a causal mask for autoregressive decoding.
            - "label": The expected output labels for the decoder, including the EOS token.
            - "src_text": The original source sentence text.
            - "tgt_text": The original target sentence text.
    """
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len) -> None:
        super().__init__() 
        self.ds = ds 
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        # self.sos_token = torch.Tensor([tokenizer_src.token_to_id('[SOS]')],dtype=torch.int64)
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)


        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')],dtype= torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')],dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds)
    

    def __getitem__(self, index) -> Any:
        src_target_pair =self.ds[index]
        
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_token = self.tokenizer_tgt.encode(tgt_text).ids 
        
        enc_num_padding_tokens = self.seq_len -len(enc_input_tokens) -2 
        dec_num_padding_tokens = self.seq_len - len(dec_input_token) -1 
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0 : 
            raise ValueError("sentence is too long")
        
        # Add SOS and EOS to source text
        encoder_input = torch.cat(
            [
            self.sos_token,
            torch.tensor(enc_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens,dtype=torch.int64)
            ]
        )
        # Add sos to decoder input
        decoder_input= torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_token,dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens,dtype=torch.int64)
            ]
        )
        # add eos to label (what we expect as output from decoder) 
        label = torch.cat(
            [
                torch.tensor(dec_input_token,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens,dtype=torch.int64)
            ]
        )
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len 
        assert label.size(0) == self.seq_len 
        
        return {
            "encoder_input" : encoder_input,#(seq_len)
            "decoder_input" : decoder_input,#(seq_len)
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_len)
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), #(1,seq_len) & (1,seq_len,seq_len)
            "label" : label, #(seq_len)
            "src_text" : src_text , 
            "tgt_text" : tgt_text
            
        }
        
def casual_mask(size): 
    
    mask = torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask == 0 

     
   