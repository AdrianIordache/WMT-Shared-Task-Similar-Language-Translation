from config_file import *

class Languages:
    def __init__(self, input_language: str = 'es', target_language: str = 'ca', preprocessing_types: List[str] = []):
        self.input_language      = input_language
        self.target_language     = target_language
        self.preprocessing_types = preprocessing_types
        
        self.dictionary = {
            input_language:  [],
            target_language: [],
        }

    def append(self, key: str, sentences: List[str]) -> None:
        assert key in self.dictionary.keys(), f"[ERROR] Only two languages allowed ({self.input_language}, {self.target_language})"
        self.dictionary[key].extend(sentences)

    def prepare_data(self, path_to_folder: str, verbose_level: bool = 1) -> None:
        if verbose_level > 0: print(f"Initial number of sentences: {len(self.dictionary[self.input_language])}")
        if os.path.exists(path_to_folder) == False: os.makedirs(path_to_folder)

        input_language_file  = open(os.path.join(path_to_folder, f'cleaned_train.{self.input_language}'), 'w')
        target_language_file = open(os.path.join(path_to_folder, f'cleaned_train.{self.target_language}'), 'w')

        cleaned_corpus = Languages.corpus_cleaning(
            corpus = list(zip(self.dictionary[self.input_language], self.dictionary[self.target_language])),
            preprocessing_types = self.preprocessing_types
        )

        removed_counter = 0
        for step, paired_senteneces in enumerate(cleaned_corpus): 
            if step % 100000 == 0 or step == len(cleaned_corpus): 
                if verbose_level > 0: print(f"[STEP]: {step}/{len(cleaned_corpus)}")

            try:
                (cleaned_input_sentence, cleaned_target_sentence) = \
                    Languages.sentences_cleaning(paired_senteneces, preprocessing_types = self.preprocessing_types)
                
                input_language_file.write(cleaned_input_sentence + '\n')
                target_language_file.write(cleaned_target_sentence + '\n')
            except:
                if verbose_level > 1: print(f"[REMOVED]: [{cleaned_input_sentence}] -> [{cleaned_target_sentence}]")
                removed_counter += 1
                continue

        corpus_final_size = len(cleaned_corpus) - removed_counter
        if verbose_level  > 0: 
            print(f"Removed {removed_counter} sentences with problems")
            print(f"Final number of sentences: {corpus_final_size}")

        input_language_file.close()
        target_language_file.close()

        with open(os.path.join(path_to_folder, f'cleaned_train.{self.input_language}') , "r", encoding = "utf-8") as src_file: src_sentences = src_file.read().splitlines()
        with open(os.path.join(path_to_folder, f'cleaned_train.{self.target_language}'), "r", encoding = "utf-8") as tgt_file: tgt_sentences = tgt_file.read().splitlines()
        
        assert len(src_sentences) == corpus_final_size or len(tgt_sentences) == corpus_final_size, "[ERROR] Number of sentences mismatch"


    def sentences_cleaning(pair: Tuple[str, str], preprocessing_types: List[str]) -> Tuple[str, str]:
        (input_sentence, target_sentence) = pair
        
        if 'langid' in preprocessing_types:
            # remove pairs for which the ro sentence is not actually in ro
            predicted_language = IDENTIFIER.classify(target_sentence)[0]
            if predicted_language != TGT_LANGUAGE:
                raise Exception('RO is not the language for this sentence')

        if 'lowercase' in preprocessing_types:
            input_sentence = input_sentence.split()
            input_sentence = [word.lower() for word in input_sentence]
            input_sentence = ' '.join(input_sentence)

            target_sentence = target_sentence.split()
            target_sentence = [word.lower() for word in target_sentence]
            target_sentence = ' '.join(target_sentence)

        return (input_sentence, target_sentence)

    def corpus_cleaning(corpus: List[Tuple[str, str]], preprocessing_types: List[str], verbose_level: int = 1) -> List[Tuple[str, str]]:
        if "drop_duplicates" in preprocessing_types:
            data = pd.DataFrame(corpus, columns = ['source_sentence', 'target_sentence'])
            initial_samples_count = data.shape[0]
            data = data.drop_duplicates()
            dropped_samples_count = initial_samples_count - data.shape[0]
            if verbose_level > 0: print(f"Removed {dropped_samples_count} duplicate sentences")
            corpus = list(zip(list(data['source_sentence'].values), list(data['target_sentence'].values)))

        return corpus

class BPEModel:
    def __init__(self, data_folder : str, vocab_size: int = 37000, min_length: int = 3, max_length : int = 100,  max_length_ratio : int = 1.5):
        with open(os.path.join(data_folder, f'cleaned_train.{SRC_LANGUAGE}'), "r", encoding = "utf-8") as src_file: src_sentences = src_file.read().splitlines()
        with open(os.path.join(data_folder, f'cleaned_train.{TGT_LANGUAGE}'), "r", encoding = "utf-8") as tgt_file: tgt_sentences = tgt_file.read().splitlines()
        
        with open(os.path.join(data_folder, f"cleaned_train.{SRC_LANGUAGE}-{TGT_LANGUAGE}"), "w", encoding = "utf-8") as output_file:
            output_file.write("\n".join(src_sentences + tgt_sentences))

        print("\n\n\nLearning BPE...")
        youtokentome.BPE.train(
            data       = os.path.join(data_folder, f"cleaned_train.{SRC_LANGUAGE}-{TGT_LANGUAGE}"), 
            vocab_size = vocab_size,
            model      = os.path.join(data_folder, f"bpe_{vocab_size}.model")
        )

        print("\n\n\nLoading BPE model...")
        bpe_model = youtokentome.BPE(model = os.path.join(data_folder, f"bpe_{vocab_size}.model"))

        print("\n\n\nFiltering...")
        pairs = list()
        removed_counter = 0
        for src_sentence, tgt_sentence in tqdm(zip(src_sentences, tgt_sentences), total = len(src_sentences)):
            src_token = bpe_model.encode(src_sentence, output_type = youtokentome.OutputType.ID)
            tgt_token = bpe_model.encode(tgt_sentence, output_type = youtokentome.OutputType.ID)
            
            len_src_token, len_tgt_token = len(src_token), len(tgt_token)

            if min_length < len_src_token < max_length and \
                    min_length < len_tgt_token < max_length and \
                    1. / max_length_ratio <= len_src_token / len_tgt_token <= max_length_ratio:
                pairs.append((src_sentence, tgt_sentence))
            else:
                removed_counter += 1
                continue

        print(f"Removed {removed_counter} sentences out of {len(src_sentences)}")
        print(f"Final number of sentences: {len(src_sentences) - removed_counter}")

        src_sentences, tgt_sentences = zip(*pairs)
        with open(os.path.join(data_folder, f"cleaned_train_filtered.{SRC_LANGUAGE}"), "w", encoding = "utf-8") as src_file: src_file.write("\n".join(src_sentences))
        with open(os.path.join(data_folder, f"cleaned_train_filtered.{TGT_LANGUAGE}"), "w", encoding = "utf-8") as tgt_file: tgt_file.write("\n".join(tgt_sentences))


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
        OBSERVATION     = "Added dropped duplicates"
        
        train_languages = Languages(
            input_language      = SRC_LANGUAGE, 
            target_language     = TGT_LANGUAGE, 
            preprocessing_types = PREPROCESSING_TYPES
        )

        train_languages = read_sources(
            sources   = DATASET_SOURCES, # train sources
            languages = train_languages
        )

        train_languages.prepare_data(
            path_to_folder = PATH_TO_DATASET,
        )

        DATASET_LOGGER.append(
            config_file = {'id': DATASET_VERSION},
            output_file = {
                'train_sources'       : DATASET_SOURCES,
                'preprocessing_types' : PREPROCESSING_TYPES,
                'observation'         : OBSERVATION
            }
        )

    if 0:
        bpe = BPEModel(
            data_folder      = PATH_TO_DATASET,
            vocab_size       = 37000,
            min_length       = 3,
            max_length       = 150,
            max_length_ratio = 2.
        )