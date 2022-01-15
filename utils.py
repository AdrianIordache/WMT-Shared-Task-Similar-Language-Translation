import gc
import re
import os
import io
import sys
import json
import math
import time
import string
import random
import pickle
import argparse
import datetime
import tempfile
import sacrebleu
import linecache
import youtokentome
import numpy as np
import pandas as pd
import sentencepiece as spm
import xml.etree.ElementTree as ET

import langid
from langid.langid import LanguageIdentifier, model

from tqdm import tqdm
from itertools import groupby
from collections import Counter
from unicodedata import normalize
from IPython.display import display
from typing import List, Iterable, Tuple, Dict

import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Transformer

from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from nltk.translate.bleu_score import corpus_bleu

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GlobalLogger:
    def __init__(self, path_to_global_logger: str, save_to_log: bool):
        self.save_to_log = save_to_log
        self.path_to_global_logger = path_to_global_logger

        if os.path.exists(self.path_to_global_logger):
            self.logger = pd.read_csv(self.path_to_global_logger)

    def append(self, config_file: Dict, output_file: Dict):
        if self.save_to_log == False: return

        if os.path.exists(self.path_to_global_logger) == False:
            config_columns = [key for key in config_file.keys()]
            output_columns = [key for key in output_file.keys()]

            columns = config_columns + output_columns 
            logger = pd.DataFrame(columns = columns)
            logger.to_csv(self.path_to_global_logger, index = False)
            
        self.logger = pd.read_csv(self.path_to_global_logger)
        sample = {**config_file, **output_file}
        columns = [key for (key, value) in sample.items()]

        row = [value for (key, value) in sample.items()]
        row = np.array(row)
        row = np.expand_dims(row, axis = 0)

        sample = pd.DataFrame(row, columns = columns)
        self.logger = self.logger.append(sample, ignore_index = True)
        self.logger.to_csv(self.path_to_global_logger, index = False)

    def get_version_id(self):
        if os.path.exists(self.path_to_global_logger) == False: return 0
        logger = pd.read_csv(self.path_to_global_logger)
        ids = logger["id"].values
        if len(ids) == 0: return 0
        return ids[-1] + 1
    
    def view(self):
        from IPython.display import display
        display(self.logger)


class Logger:
    def __init__(self, path_to_logger: str = 'logger.log', distributed = False):
        from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler

        self.logger = getLogger(__name__)
        self.logger.setLevel(INFO)

        if distributed == False:
            handler1 = StreamHandler()
            handler1.setFormatter(Formatter("%(message)s"))
            self.logger.addHandler(handler1)

        handler2 = FileHandler(filename = path_to_logger)
        handler2.setFormatter(Formatter("%(message)s"))
        self.logger.addHandler(handler2)

    def print(self, message):
        self.logger.info(message)

    def close(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)


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


class CrossEntropyLossSmoothed(torch.nn.Module):
    def __init__(self, eps: int = 0.1):
        super(CrossEntropyLossSmoothed, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, lengths):
        """
        :param inputs:  hypothesis target language sequences, a tensor of size (N, pad_length, vocab_size)
        :param targets: reference  target language sequences, a tensor of size (N, pad_length)
        :param lengths: true lengths of these sequences, to be able to ignore pads, a tensor of size (N)
        :return: mean label-smoothed cross-entropy loss, a scalar
        """

        # Remove pad-positions and flatten 
        # (sum(lengths), vocab_size)
        inputs, _, _, _ = pack_padded_sequence(
            input          = inputs,
            lengths        = lengths,
            batch_first    = True,
            enforce_sorted = False
        ) 

        # (sum(lengths))
        targets, _, _, _ = pack_padded_sequence(
            input          = targets,
            lengths        = lengths,
            batch_first    = True,
            enforce_sorted = False
        )

        # "Smoothed" one-hot vectors for the gold sequences
        target_vector = torch.zeros_like(inputs).scatter(dim = 1, index = targets.unsqueeze(1), value = 1.).to(DEVICE)  # (sum(lengths), n_classes), one-hot
        target_vector = target_vector * (1. - self.eps) + self.eps / target_vector.size(1)  # (sum(lengths), n_classes), "smoothed" one-hot
        
        # Compute smoothed cross-entropy loss
        loss = (-1 * target_vector * F.log_softmax(inputs, dim=1)).sum(dim=1)  # (sum(lengths))
        
        # Compute mean loss
        loss = torch.mean(loss)
        return loss


def get_positional_encoding(d_model: int, max_seq_len: int = 100):
    positional_encoding = torch.zeros((max_seq_len, d_model)) 
    for i in range(max_seq_len):
        for j in range(d_model):
            if j % 2 == 0:
                positional_encoding[i, j] = math.sin(i / math.pow(10000, j / d_model))
            else:
                positional_encoding[i, j] = math.cos(i / math.pow(10000, (j - 1) / d_model))

    positional_encoding = positional_encoding.unsqueeze(0)  # (1, max_seq_len, d_model)
    return positional_encoding


def get_lr(step, d_model, warmup_steps):
    lr = 2. * math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))
    return lr


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
