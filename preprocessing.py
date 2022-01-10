from utils import *

class Languages:
    def __init__(self, input_language: str = 'es', target_language: str = 'ca', preprocessing_types: List[str] = []):
        self.input_language      = input_language
        self.target_language     = target_language
        self.preprocessing_types = preprocessing_types

        self.dictionary = {
            input_language:  [],
            target_language: [],
        }

    def append(self, key, sentences):
        assert key in self.dictionary.keys(), "Only two languages"
        self.dictionary[key].extend(sentences)

    def preprocessing(self, path_to_output: str, data_type: str = 'train', data_version: int = 0, verbose: bool = False):
        path_to_folder = os.path.join(path_to_output, f"version-{data_version}")
        if os.path.exists(path_to_folder) == False: os.makedirs(path_to_folder)

        input_language_file  = open(os.path.join(path_to_folder, f'cleaned_{data_type}.{self.input_language}'), 'w')
        target_language_file = open(os.path.join(path_to_folder, f'cleaned_{data_type}.{self.target_language}'), 'w')

        for step, paired_senteneces in enumerate(zip(self.dictionary[self.input_language], \
                                                                     self.dictionary[self.target_language])): 

            if step % 100000 == 0 or step == len(self.dictionary[self.input_language]): 
                print(f"[STEP]: {step}/{len(self.dictionary[self.input_language])}")

            try:
                (cleaned_input_sentence, cleaned_target_sentence) = \
                    Languages.text_cleaning(paired_senteneces, preprocessing = self.preprocessing_types)
                
                input_language_file.write(cleaned_input_sentence + '\n')
                target_language_file.write(cleaned_target_sentence + '\n')
            except:
                if verbose: print(f"[REMOVED]: [{input_sentence}] -> [{target_sentence}]")
                continue

        input_language_file.close()
        target_language_file.close()

    def text_cleaning(pair: Tuple[str, str], preprocessing: List[str]):
        (input_sentence, target_sentence) = pair
        
        if 'langid' in preprocessing:
            # remove pairs for which the ro sentence is not actually in ro
            predicted_language = identifier.classify(target_sentence)[0]
            if predicted_language != TGT_LANGUAGE:
                raise Exception('RO is not the language for this sentence')

        if 'lowercase' in preprocessing:
            input_sentence = input_sentence.split()
            input_sentence = [word.lower() for word in input_sentence]
            input_sentence = ' '.join(input_sentence)

            target_sentence = target_sentence.split()
            target_sentence = [word.lower() for word in target_sentence]
            target_sentence = ' '.join(target_sentence)

        # table    = str.maketrans('', '', string.punctuation)
        # re_print = re.compile('[^%s]' % re.escape(string.printable))
        
        # sentence = normalize('NFD', sentence).encode('ascii', 'ignore')
        # sentence = sentence.decode('UTF-8')
        # sentence = sentence.split()

        # lowercase
        # sentence = [word.lower() for word in sentence]
        # sentence = [word.translate(table) for word in sentence]
        # sentence = [re_print.sub('', w) for w in sentence]
        # sentence = [word for word in sentence if word.isalpha()]
        # sentence = ' '.join(sentence)

        return (input_sentence, target_sentence)

def read_source_one(path_to_source: str, languages: Languages) -> Languages:
    for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
        path = os.path.join(path_to_source, f'Europarl.es-ro.{language}')
        with open(path, 'r') as file: lines = file.read().splitlines()
        languages.append(language, lines)

    return languages

def read_source_two(path_to_source: str, languages: Languages) -> Languages:
    path  = os.path.join(path_to_source, 'wikititles-v3.es-ro.tsv')
    lines = pd.read_csv(path, sep = '\t')
    src_sentences = lines.iloc[:, 0].values.tolist()
    tgt_sentences = lines.iloc[:, 1].values.tolist()
    languages.append(SRC_LANGUAGE, src_sentences)
    languages.append(TGT_LANGUAGE, tgt_sentences)

    return languages

def read_source_three(path_to_source: str, languages: Languages) -> Languages:
    for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
        path = os.path.join(path_to_source, f'TildeMODEL.es-ro.{language}')
        with open(path, 'r') as file: lines = file.read().splitlines()
        languages.append(language, lines)

    return languages

def read_source_four(path_to_source: str, languages: Languages) -> Languages:
    for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
        path = os.path.join(path_to_source, f'JRC-Acquis.es-ro.{language}')
        with open(path, 'r') as file: lines = file.read().splitlines()
        languages.append(language, lines)

    return languages

def read_source_dev(path_to_source: str, languages: Languages) -> Languages:
    for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
        path = os.path.join(path_to_source, f'dev.ro-es.{language}')
        with open(path, 'r') as file: lines = file.read().splitlines()
        languages.append(language, lines)

    return languages

def read_sources(sources: List[int], languages: Languages) -> Languages:
    if 1 in sources:    
        languages = read_source_one(PATH_TO_SOURCE_1, languages)

    if 2 in sources:
        languages = read_source_two(PATH_TO_SOURCE_2, languages)

    if 3 in sources:
        languages = read_source_three(PATH_TO_SOURCE_3, languages)

    if 4 in sources:
        languages = read_source_four(PATH_TO_SOURCE_4, languages)

    if "dev" in sources:
        languages = read_source_dev(PATH_TO_SOURCE_DEV, languages)

    return languages

if __name__ == '__main__':
    if 0:
        print("Train Preprocessing...")
        train_languages = Languages(SRC_LANGUAGE, TGT_LANGUAGE, PREPROCESSING_METHODS)
        train_languages = read_sources(
            sources   = [1, 2, 3, 4], # train sources
            languages = train_languages
        )

        train_languages.preprocessing(
            path_to_output = 'data/cleaned/',
            data_type      = 'train',
            data_version   = DATASET_VERSION
        )

    if 0: 
        print("Valid Preprocessing...")
        dev_languages = Languages(SRC_LANGUAGE, TGT_LANGUAGE, [])
        dev_languages = read_sources(
            sources   = ['dev'], # valid sources
            languages = dev_languages
        )

        dev_languages.preprocessing(
            path_to_output = 'data/cleaned/',
            data_type      = 'valid',
            data_version   = DATASET_VERSION
        )


    if 1:
        path_to_folder = os.path.join(f"models/version-{DATASET_VERSION}")
        if os.path.exists(path_to_folder) == False: os.makedirs(path_to_folder)

        spm.SentencePieceTrainer.train(
            input                  = [f'data/cleaned/version-{DATASET_VERSION}/cleaned_train.es', f'data/cleaned/version-{DATASET_VERSION}/cleaned_train.ro'],
            model_prefix           = os.path.join(path_to_folder, 'sentpiece_32k'),
            character_coverage     = 1,
            vocab_size             = 32000,
            shuffle_input_sentence = True,
            unk_id                 = UNK_IDX,
            bos_id                 = BOS_IDX,
            eos_id                 = EOS_IDX,
            pad_id                 = PAD_IDX,
            unk_piece              = '<unk>',
            bos_piece              = '<bos>',
            eos_piece              = '<eos>',
            pad_piece              = '<pad>',
        )
