from utils import *

DATASET_VERSION = 3
DATASET_LOGGER  = GlobalLogger(path_to_global_logger = f'data/cleaned/dataset_logger.csv', save_to_log = True)

DECIMALS     = 4
SEED         = 42
SRC_LANGUAGE = 'es'
TGT_LANGUAGE = 'ro'

PATH_TO_DATA       = 'data/'
PATH_TO_SOURCE_1   = os.path.join(PATH_TO_DATA, 'source-1')
PATH_TO_SOURCE_2   = os.path.join(PATH_TO_DATA, 'source-2')
PATH_TO_SOURCE_3   = os.path.join(PATH_TO_DATA, 'source-3')
PATH_TO_SOURCE_4   = os.path.join(PATH_TO_DATA, 'source-4')
PATH_TO_SOURCE_DEV = os.path.join(PATH_TO_DATA, 'source-dev')

IDENTIFIER          = LanguageIdentifier.from_modelstring(model, norm_probs = True)
DATASET_SOURCES     = [1, 2, 3, 4]
PREPROCESSING_TYPES = ['langid', 'lowercase', 'drop_duplicates']

PATH_TO_LOG       = os.path.join('logs', f'dataset-{DATASET_VERSION}')
PATH_TO_MODELS    = os.path.join('models', f'dataset-{DATASET_VERSION}')
PATH_TO_DATASET   = os.path.join('data', 'cleaned', f'dataset-{DATASET_VERSION}')

VOCAB_SIZE        = 16000
PATH_TO_BPE_MODEL = f"bpe/dataset-{DATASET_VERSION}/bpe_{VOCAB_SIZE}.model"

PATH_TO_DATASET_FILES = {
	'train': {
		SRC_LANGUAGE: os.path.join(PATH_TO_DATA, 'cleaned', f'dataset-{DATASET_VERSION}', f'cleaned_train_filtered_{VOCAB_SIZE}.es'),
		TGT_LANGUAGE: os.path.join(PATH_TO_DATA, 'cleaned', f'dataset-{DATASET_VERSION}', f'cleaned_train_filtered_{VOCAB_SIZE}.ro')
	},	 

	'valid': {
		SRC_LANGUAGE: os.path.join(PATH_TO_DATA, 'source-dev', 'dev.ro-es.es'),
    	TGT_LANGUAGE: os.path.join(PATH_TO_DATA, 'source-dev', 'dev.ro-es.ro')
	},

	'test': {
		SRC_LANGUAGE: os.path.join(PATH_TO_DATA, 'source-test', 'test.es-ro.es'),
    	TGT_LANGUAGE: os.path.join(PATH_TO_DATA, 'source-test', 'test-ref.es-ro.ro')
	}
}

RD     = lambda x: np.round(x, DECIMALS)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

OUTPUT = {
    'train_loss': None,
    'valid_loss': None,
    'test_loss':  None,

    'valid_blue_score': None,
    'test_blue_score': None
}

seed_everything(SEED)

