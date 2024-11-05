import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import warnings
from pathlib import Path
from tqdm import tqdm

from dataset import BilingualDataset, causal_mask
from model import build_transfomer
from config import get_weights_file_path, get_config

def get_all_sentences(ds, lang):
    """
    Each item is a pair of sentences: english and non-english pairs.
    This function returns the sentences in the specified language.
    """
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    Tokenizer is used to convert the words into tokens.
    These tokens are smaller units of the sentence/corpus.
    These tokens are then converted into numbers which are used to train the model.

    Args:
        config: dictionary with the following
        ds: dataset
        lang: language
    """
    # Eg: config['tokenizer_file'] = '../tokenizer/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # build the tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))  # If the word is not in the vocabulary, replace it with [UNK]
        tokenizer.pre_tokenizer = Whitespace()  # splitting the words based on whitespaces
        # unknown, padding, start of sentence, end of sentence
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)  # min_frequency=2 means that the word should appear at least 2 times in the dataset
        # trainer is used to train the tokenizer on the dataset. It will learn the vocabulary and the frequency of the words
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    """
    Load the dataset from the Hugging Face datasets library.
    """
    ds_raw = load_dataset("opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # build the tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # split train into train and validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, 
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, 
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_len_src, max_len_tgt = 0, 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src},\nMax length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)

    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)  # want to process each sentence one by one

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transfomer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config):
    # define the device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Device used for training: ", device)
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Starting the tensorboard for visualization of the loss
    writer = SummaryWriter(log_dir=config['experiment_name'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)

    inital_epoch = 0
    global_step = 0
    if config['preload_model']:
        model_filename = get_weights_path(config, config['preload_model'])
        print(f"Preloading model {model_filenme}")
        state = torch.load(model_filename)
        inital_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # loss function is cross entropy and we use ignore index to ignore the padding token
    # label smoothing is used for regularization to avoid overfitting
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # training loop
    for epoch in range(inital_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Training epoch: {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            
            # run the tensors thorugh the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (B, seq_len)
            # (B, seq_len, tgt_vocab_size) --> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            batch_iterator.set_postfix({f"Loss": f"{loss.item():6.3f}"})

            # write the loss to tensorboard
            writer.add_scalar('train/loss', loss.item(), global_step)

            # backpropogate the loss and update the model parameters
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        
        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)

