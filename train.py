import torch 
import torch.nn as nn 

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path 
from torch.utils.data import DataLoader,Dataset,random_split



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
    Loads a dataset and splits it into training and validation sets, also builds tokenizers for source and target languages.

    Parameters:
    - config (dict): Configuration dictionary that must include 'lang_src' and 'lang_tgt' keys for source and target languages,
                        and 'tokenizer_file' template for saving or loading the tokenizer.

    Returns:
    - tuple: A tuple containing two elements, the training dataset and the validation dataset, both split from the original dataset.

    This function performs the following steps:
    1. Loads the dataset specified by 'opus_books' and the language pair from the config.
    2. Builds or loads a tokenizer for both the source and target languages using the `get_or_build_tokenizer` function.
    3. Splits the dataset into training (90%) and validation (10%) sets.
    4. Returns the training and validation datasets.

    Note: This function requires the 'datasets' and 'torch' libraries, and it assumes the existence of a function named `get_or_build_tokenizer`
    that is capable of either loading an existing tokenizer or training a new one based on the dataset provided.
    """
    
    ds_raw = load_dataset("opus_books",f'{config["lang_src"]} - {config['lang_tgt']}',split="train")
    
    # Build tokenizer 
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])
    
    # Keep 90% for training and 10% for validation 
    train_ds_size = int(0.9 * len(ds_raw))
    valid_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, valid_ds_raw = random_split(ds_raw, [train_ds_size, valid_ds_size])
        
        # return train_ds_raw, valid_ds_raw
    
    