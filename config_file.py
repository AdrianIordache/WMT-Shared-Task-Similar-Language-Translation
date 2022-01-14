from utils import *

SRC_LANGUAGE = 'es'
TGT_LANGUAGE = 'ro'

PATH_TO_DATA       = 'data/'
PATH_TO_SOURCE_1   = os.path.join(PATH_TO_DATA, 'source-1')
PATH_TO_SOURCE_2   = os.path.join(PATH_TO_DATA, 'source-2')
PATH_TO_SOURCE_3   = os.path.join(PATH_TO_DATA, 'source-3')
PATH_TO_SOURCE_4   = os.path.join(PATH_TO_DATA, 'source-4')
PATH_TO_SOURCE_DEV = os.path.join(PATH_TO_DATA, 'source-dev')

DATASET_LOGGER      = GlobalLogger(path_to_global_logger = f'data/cleaned/dataset_logger.csv', save_to_log = True)
DATASET_VERSION     = DATASET_LOGGER.get_version_id() - 1
IDENTIFIER          = LanguageIdentifier.from_modelstring(model, norm_probs = True)
DATASET_SOURCES     = [1, 2, 3, 4]
PREPROCESSING_TYPES = ['langid', 'lowercase', 'drop_duplicates']

PATH_TO_LOG       = os.path.join('logs', f'dataset-{DATASET_VERSION}')
PATH_TO_MODELS    = os.path.join('models', f'dataset-{DATASET_VERSION}')
PATH_TO_DATASET   = os.path.join('data', 'cleaned', f'dataset-{DATASET_VERSION}')

PATH_TO_CLEANED_TRAIN = {
    SRC_LANGUAGE: os.path.join(PATH_TO_DATA, 'cleaned', f'dataset-{DATASET_VERSION}', 'cleaned_train_filtered.es'),
    TGT_LANGUAGE: os.path.join(PATH_TO_DATA, 'cleaned', f'dataset-{DATASET_VERSION}', 'cleaned_train_filtered.ro')
} 

PATH_TO_CLEANED_VALID = {
    SRC_LANGUAGE: os.path.join(PATH_TO_DATA, 'cleaned', f'dataset-{DATASET_VERSION}', 'cleaned_dev_filtered.es'),
    TGT_LANGUAGE: os.path.join(PATH_TO_DATA, 'cleaned', f'dataset-{DATASET_VERSION}', 'cleaned_dev_filtered.ro')
} 

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']

DECIMALS  = 4
SEED      = 42
RD        = lambda x: np.round(x, DECIMALS)
DEVICE    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

OUTPUT = {
    'train_loss': None,
    'valid_loss': None,
    'test_loss':  None,

    'valid_blue_score': None,
    'test_blue_score': None
}

seed_everything(SEED)