import torch 
import torch.nn as nn 

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path 
from torch.utils.data import DataLoader,Dataset,random_split

from dataset import BilingualDataset,casual_mask
from model import build_transformer

def get_or_build_tokenizer(config, ds, lang):
    """
    Retrieves or builds a tokenizer based on the given language and dataset.

    Parameters:
    - config (dict): Configuration dictionary containing the path template for the tokenizer file.
    - ds: Dataset from which sentences will be extracted for tokenizer training.
    - lang (str): Language code to specify the language for the tokenizer.

    Returns:
    - Tokenizer: An instance of the Tokenizer class, either loaded from a pre-existing file or trained and saved during the function call.

    This function first checks if a tokenizer file for the specified language exists at the path defined in the config dictionary.
    If the file does not exist, it creates a new WordLevel tokenizer, trains it on sentences extracted from the provided dataset,
    and saves the tokenizer to the specified path. If the file exists, it loads the tokenizer from the file.
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer(Whitespace)
        trainer = WordLevelTrainer(special_tokens = ["[UNK]","[PAD]","[EOS]","[SOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentence(ds,lang),trainer=trainer) 
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_all_sentence(ds,lang):
    for item in ds: 
        yield item['translation'][lang]

def get_ds(config):
    
    """
    Prepares training and validation dataloaders for a bilingual dataset, along with source and target tokenizers.

    Parameters:
    - config (dict): Configuration dictionary with keys including 'lang_src' (source language code), 
      'lang_tgt' (target language code), 'seq_len' (sequence length for padding/truncation), 
      and 'batch_size' (batch size for the dataloaders).

    Returns:
    - Tuple containing:
        - train_dataloader (DataLoader): DataLoader for the training dataset.
        - val_dataloader (DataLoader): DataLoader for the validation dataset.
        - tokenizer_src (Tokenizer): Tokenizer for the source language.
        - tokenizer_tgt (Tokenizer): Tokenizer for the target language.

    This function loads a bilingual dataset from the "opus_books" collection, splits it into training and validation sets,
    and prepares DataLoaders for each. It also reports the maximum sequence lengths found in the raw dataset for both source
    and target languages. Tokenizers for both languages are either loaded or trained as needed.
    """
    
    ds_raw = load_dataset("opus_books",f'{config["lang_src"]} - {config['lang_tgt']}',split="train")
    
    # Build tokenizer 
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])
    
    # Keep 90% for training and 10% for validation 
    train_ds_size = int(0.9 * len(ds_raw))
    valid_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, valid_ds_raw = random_split(ds_raw, [train_ds_size, valid_ds_size])
    
    
    train_ds = BilingualDataset(train_ds_raw,tokenizer_src, tokenizer_tgt, config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds = BilingualDataset(valid_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    
    max_len_src= 0 
    max_len_tgt= 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids 
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids 
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))
        
    print(f"Max length of source sentence {max_len_src}")
    print(f"Max length of target sentence {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    val_dataloader = DataLoader(val_ds,batch_size=1,shuffle=True)
    
    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt


def get_model(config, vocab_src_len,vocab_tgt_len):
    model = build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model