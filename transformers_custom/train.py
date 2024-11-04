import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from .dataset import BilingualDataset, causal_mask
from .model import build_transfomer

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
    # yet to be implemented
    pass






