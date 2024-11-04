import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self,
                 ds,
                 tokenizrer_src,
                 tokenizer_tgt,
                 src_lang,
                 tgt_lang,
                 seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizrer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # converting the special tokens to tensors IDs
        # can use any tokenizer: src or tgt
        self.sos_token = torch.Tensor(tokenizrer_src.token_to_id('[SOS]'), dtype=torch.int64)
        self.eos_token = torch.Tensor(tokenizrer_src.token_to_id('[EOS]'), dtype=torch.int64)
        self.pad_token = torch.Tensor(tokenizrer_src.token_to_id('[PAD]'), dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # convert the text to tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids  # gives an array of token IDs
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len = len(enc_input_tokens) - 2  # subtracting 2 because of [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len = len(dec_input_tokens) - 1  # subtracting 1 because of [SOS]; dec doesnt have [EOS]

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('The sequence length is too short')
        
        # two tensors for encoder input and decoder input
        # add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token, 
                torch.Tensor(enc_input_tokens, dtype=torch.int64), 
                self.eos_token,
                torch.Tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)  # padding tokens added to have the same length sentences as input every time

            ])
        
        # add SOS to the target text
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(dec_input_tokens, dtype=torch.int64),
                torch.Tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ])
        
        # add EOS to the lavbel/target
        label = torch.cat(
            [
                torch.Tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.Tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ])
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len,)
            "decoder_input": decoder_input, # (seq_len,)
            # (seq_len,) -> (1, seq_len) -> (1, 1, seq_len)
            # this is used because we dont want self attention mechanism to attend to the padding tokens
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # unsqueeze(0) adds a dimension at the beginning
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,  # (seq_len,)
            "src_txt": src_text,
            "tgt_txt": tgt_text
            }

def causal_mask(size):
    """
    This function is used to create a mask that will be used to mask the future tokens.
    This is because we only want the model to look at current and previous words and not future words.
    The model should only attend to the tokens that are before the current token.

    The first line will return all the values above the diagonal. Everything below the diag becomes 0
    But we want only the lower triangle. Hence, we use == 0 to get the lower triangle.
    """

    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)  # upper triangular matrix
    return mask == 0  # 0 means that the model should not attend to that token

