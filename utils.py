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

from typing import List, Iterable
from collections import Counter
from unicodedata import normalize
from IPython.display import display

SRC_LANGUAGE = 'es'
TGT_LANGUAGE = 'ro'

PATH_TO_DATA       = 'data/'
PATH_TO_SOURCE_1   = os.path.join(PATH_TO_DATA, 'source-1')
PATH_TO_SOURCE_2   = os.path.join(PATH_TO_DATA, 'source-2')
PATH_TO_SOURCE_3   = os.path.join(PATH_TO_DATA, 'source-3')
PATH_TO_SOURCE_4   = os.path.join(PATH_TO_DATA, 'source-4')
PATH_TO_SOURCE_DEV = os.path.join(PATH_TO_DATA, 'source-dev')


