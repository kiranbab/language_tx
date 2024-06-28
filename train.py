import torch 
import torch.nn as nn 

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path 
from torch.utils.data import DataLoader,Dataset,random_split
from tqdm import tqdm

import warnings
from dataset import BilingualDataset,casual_mask
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from config import get_weights_file_path,get_config


def run_validation(model,validation_ds,tokenizer_src,tokenizer_tgt,max_len,device,print_msg,global_state,writer,num_examples =2):
    model.eval()

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
        tokenizer.pre_tokenizer=Whitespace()
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
    
    ds_raw = load_dataset("opus_books",f'{config["lang_src"]}-{config["lang_tgt"]}',split="train")
    
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

def train_model(config):
    # define the device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    print(f"Using device {device}")
    
    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)
    
    train_dataloader,val_dataloader, tokenizer_src,tokenizer_tgt = get_ds(config)
    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard 
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)
    
    intial_epoch = 0 
    global_step = 0 
    if config['preload']:
        model_filename = get_weights_file_path(config,config['preload'])
        print(f"Pre-Loading model {model_filename}")
        state = torch.load(model_filename)
        intial_epoch = state['epoch']+1 
        optimizer.load_state_dict(state['load_state_dict'])
        global_step = state['global_step']
        
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),label_smoothing=0.1).to(device)
    
    for epoch in range(intial_epoch,config['num_epochs']):
        model.train() 
        batch_iterator = tqdm(train_dataloader,desc= 'Processing epoch : {epoch:02d}')
        for batch in batch_iterator:
            encoder_input  =batch['encoder_input'].to(device) #(B,seq_len)
            decoder_input =batch['decoder_input'].to(device) #(B,seq_len)
            encoder_mask = batch['encoder_mask'].to(device) #(B,1,1,seq_len)
            decoder_mask = batch['decoder_mask'].to(device) #(B,1,seq_len,seq_len)
            
            # Run the tensor through the transformer 
            encoder_output = model.encode(encoder_input,encoder_mask) #(B,seq_len,d_model)
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask) # (B,seq_len,d_mdoel)
            proj_output = model.projection_layer(decoder_output) # (B,seq_len,target_vocab_size)
            
            label = batch['label'].to(device) #(B,seq_len) 
            
            #(B,seq_len,tgt_vocab_size) -> (B*seq_len, tgt_vocab_size) 
            loss = loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1)) 
            batch_iterator.set_postfix({f"loss" : f"{loss.item():6.3f}"})
            
            # Log the loss 
            writer.add_scalar('train loss',loss.item(),global_step)
            writer.flush () 
            
            # Backpropagate the loss 
            loss.backward() 
            
            # update the weights 
            optimizer.step() 
            optimizer.zero_grad() 
            
            global_step +=1 
            
        # save the model at every epoch 
        model_filename = get_weights_file_path(config,f'{epoch:02d}')
        torch.save(
            {
             'epoch': epoch ,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict' : optimizer.state_dict(),
             'global_step' : global_step,  
            },model_filename)
        
    
if __name__ == '__main__':
    warnings.filterwarnings('ignore')     
    config = get_config()
    train_model(config)
           
            
    