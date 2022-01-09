import gc
import re
import os
import io
import math
import time
import string
import random
import pickle
import tempfile
import linecache
import numpy as np
import pandas as pd
import sentencepiece as spm
import xml.etree.ElementTree as ET

import langid
from langid.langid import LanguageIdentifier, model

from typing import List, Iterable, Tuple
from collections import Counter
from unicodedata import normalize
from IPython.display import display

import torch
import torchtext
import torch.nn as nn

from torch import Tensor
from torch.nn import Transformer

from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SRC_LANGUAGE = 'es'
TGT_LANGUAGE = 'ro'

PATH_TO_DATA       = 'data/'
PATH_TO_SOURCE_1   = os.path.join(PATH_TO_DATA, 'source-1')
PATH_TO_SOURCE_2   = os.path.join(PATH_TO_DATA, 'source-2')
PATH_TO_SOURCE_3   = os.path.join(PATH_TO_DATA, 'source-3')
PATH_TO_SOURCE_4   = os.path.join(PATH_TO_DATA, 'source-4')
PATH_TO_SOURCE_DEV = os.path.join(PATH_TO_DATA, 'source-dev')

identifier = LanguageIdentifier.from_modelstring(model, norm_probs = True)
PREPROCESSING_METHODS = ['langid', 'lowercase']
DATASET_VERSION = 2

PATH_TO_MODEL = os.path.join('models', f'version-{DATASET_VERSION}', 'sentpiece.model')

PATH_TO_CLEANED_TRAIN = {
    SRC_LANGUAGE: os.path.join(PATH_TO_DATA, 'cleaned', f'version-{DATASET_VERSION}', 'cleaned_train.es'),
    TGT_LANGUAGE: os.path.join(PATH_TO_DATA, 'cleaned', f'version-{DATASET_VERSION}', 'cleaned_train.ro')
} 

PATH_TO_CLEANED_VALID = {
    SRC_LANGUAGE: os.path.join(PATH_TO_DATA, 'cleaned', f'version-{DATASET_VERSION}', 'cleaned_valid.es'),
    TGT_LANGUAGE: os.path.join(PATH_TO_DATA, 'cleaned', f'version-{DATASET_VERSION}', 'cleaned_valid.ro')
} 

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']

SEED      = 42
DECIMALS  = 4
VERBOSITY = 50
RD        = lambda x: np.round(x, DECIMALS)
DEVICE    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def seed_everything(SEED = 42):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator()
    generator.manual_seed(SEED)

def seed_worker(worker_id):
    np.random.seed(SEED)
    random.seed(SEED)

seed_everything(SEED)

def time_since(since, percent):
    def seconds_as_minutes(seconds):
        import math
        minutes  = math.floor(seconds / 60)
        seconds -= minutes * 60
        return f'{int(minutes)}m {int(seconds)}s'

    now     = time.time()
    seconds = now - since

    total_seconds    = seconds / (percent)
    remained_seconds = total_seconds - seconds
    return f'{seconds_as_minutes(seconds)} (remain {seconds_as_minutes(remained_seconds)})'

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def update(self, value, n = 1):
        self.count += n
        self.sum   += value * n

        self.value   = value 
        self.average = self.sum / self.count

    def reset(self):
        self.value, self.average, self.sum, self.count = 0, 0, 0, 0

def generate_batch(data_batch):
    src_batch, tgt_batch = [], []
        
    for step, (src_encodings, tgt_encodings) in enumerate(data_batch):
        src_batch.append(
            torch.cat(
                (torch.tensor([BOS_IDX]), src_encodings, torch.tensor([EOS_IDX])), dim = 0
            )
        )

        tgt_batch.append(
            torch.cat(
                (torch.tensor([BOS_IDX]), tgt_encodings, torch.tensor([EOS_IDX])), dim = 0
            )
        )

    src_batch = pad_sequence(src_batch, padding_value = PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value = PAD_IDX)

    return src_batch, tgt_batch

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device = DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device = DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def free_gpu_memory(device, object = None, verbose = False):
    if object == None:
        for object in gc.get_objects():
            try:
                if torch.is_tensor(object) or (hasattr(object, 'data' and  torch.is_tensor(object.data))):
                    if verbose: print(f"GPU Memory Used: {object}, with size: {object.size()}")
                    object = object.detach().cpu()
                    del object
            except:
                pass
    else:
        object = object.detach().cpu()
        del object

    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

def train_epoch(model, loader, optimizer, loss_fn, epoch, CFG):
    model.train()
    losses_plot = []

    losses      = AverageMeter()
    start = end = time.time()

    for step, (src, tgt) in enumerate(loader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss    = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        loss.backward()
        losses.update(loss.item(), CFG['batch_size_t'])
        
        optimizer.step()
        end = time.time()

        if step % CFG['print_freq'] == 0 or step == (len(loader) - 1):
            print('[GPU {0}][TRAIN] Epoch: [{1}][{2}/{3}], Elapsed {remain:s}, Loss: {loss.value:.3f}({loss.average:.3f})'
                  .format(DEVICE, epoch + 1, step, len(loader), 
                    remain   = time_since(start, float(step + 1) / len(loader)), 
                    loss     = losses)
            )

        losses_plot.append(losses.value)

    free_gpu_memory(DEVICE)
    return losses.average, np.mean(losses_plot)


def valid_epoch(model, loader, loss_fn, CFG):
    model.eval()

    losses_plot = []
    losses      = AverageMeter()
    start = end = time.time()

    for step, (src, tgt) in enumerate(loader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss    = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        losses.update(loss.item(), CFG['batch_size_v'])
        end = time.time()

        if step % CFG['print_freq'] == 0 or step == (len(loader) - 1):
            print('[GPU {0}][VALID] Epoch: [[{1}/{2}], Elapsed {remain:s}, Loss: {loss.value:.3f}({loss.average:.3f})'
                  .format(DEVICE, step, len(loader), 
                    remain   = time_since(start, float(step + 1) / len(loader)), 
                    loss     = losses)
            )

        losses_plot.append(losses.value)

    free_gpu_memory(DEVICE)
    return losses.average, np.mean(losses_plot)
