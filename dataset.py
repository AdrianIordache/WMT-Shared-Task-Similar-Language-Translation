from config_file import *

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, sentpiece_model):
        super().__init__()

        self.src_sentences   = src_sentences
        self.tgt_sentences   = tgt_sentences
        self.sentpiece_model = sentpiece_model 

        self.sp = spm.SentencePieceProcessor(model_file = self.sentpiece_model)

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence, tgt_sentence = False, False
          
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        src_sentence_encoded = self.sp.encode(src_sentence.strip(), out_type = int)
        tgt_sentence_encoded = self.sp.encode(tgt_sentence.strip(), out_type = int)

        src_tensor = torch.tensor(src_sentence_encoded, dtype = torch.long)
        tgt_tensor = torch.tensor(tgt_sentence_encoded, dtype = torch.long)
        
        return src_tensor, tgt_tensor

class SequenceLoader(object):
    def __init__(self, data_folder, vocab_size, source_language, target_language, dataset_type, tokens_in_batch):
        self.tokens_in_batch = tokens_in_batch
        self.source_language = source_language
        self.target_language = target_language

        assert dataset_type.lower() in ["train", "valid", "test"], \
            "'split' must be in ['train', 'valid', 'test']"
        
        self.dataset_type   = dataset_type.lower()
        self.training_stage = self.dataset_type == "train"
        self.bpe_model      = youtokentome.BPE(model = PATH_TO_BPE_MODEL)

        with open(PATH_TO_DATASET_FILES[self.dataset_type][SRC_LANGUAGE], "r", encoding = "utf-8") as src_file:
            source_sentences = src_file.read().splitlines()

        with open(PATH_TO_DATASET_FILES[self.dataset_type][TGT_LANGUAGE], "r", encoding = "utf-8") as tgt_file:
            target_sentences = tgt_file.read().splitlines()

        assert len(source_sentences) == len(target_sentences), "[ERROR] Number of sentences mismatch"

        source_lengths = [len(src) for src in \
            self.bpe_model.encode(source_sentences, bos = False, eos = False)]
        target_lengths = [len(tgt) for tgt in \
            self.bpe_model.encode(target_sentences, bos = True,  eos = True )]  # target language sequences have <BOS> and <EOS> tokens

        self.data = list(zip(source_sentences, target_sentences, source_lengths, target_lengths))

        # If for training, pre-sort by target lengths - required for itertools.groupby() later
        if self.training_stage: self.data.sort(key = lambda sample: sample[3])

        self.generate_batches()

    def generate_batches(self):

        if self.training_stage:
            # Group or chunk based on target sequence lengths
            # There will be a list of tuples containing only sequences (tgt strings) of certain size, for speed?
            chunks = [list(group) for group_len, group in groupby(self.data, key = lambda sample: sample[3])]
            
            self.batches = list()
            for chunk in chunks:
                # Sort inside chunk by source sequence lengths, so that a batch would also have similar source sequence lengths
                chunk.sort(key = lambda sample: sample[2])
                seqs_per_batch = self.tokens_in_batch // chunk[0][3]
                self.batches.extend([chunk[i : i + seqs_per_batch] for i in range(0, len(chunk), seqs_per_batch)])
                # break

            random.shuffle(self.batches)
            self.n_batches = len(self.batches)
            self.current_batch = -1
        else:
            self.batches       = [[chunk] for chunk in self.data]
            self.n_batches     = len(self.batches)
            self.current_batch = -1


    def __iter__(self):
        return self

    def __next__(self):
        self.current_batch += 1
        
        try:
            source_sentences, target_sentences, source_lengths, target_lengths = zip(*self.batches[self.current_batch])
        except IndexError:
            raise StopIteration

        source_sentences = self.bpe_model.encode(source_sentences, output_type = youtokentome.OutputType.ID, bos = False, eos = False)
        target_sentences = self.bpe_model.encode(target_sentences, output_type = youtokentome.OutputType.ID, bos = True,  eos = True)

        source_sentences = pad_sequence(
                sequences     = [torch.LongTensor(sentences) for sentences in source_sentences],
                batch_first   = True,
                padding_value = self.bpe_model.subword_to_id('<PAD>')
        )

        target_sentences = pad_sequence(
                sequences     = [torch.LongTensor(sentences) for sentences in target_sentences],
                batch_first   = True,
                padding_value = self.bpe_model.subword_to_id('<PAD>')
        )

        source_lengths = torch.LongTensor(source_lengths)
        target_lengths = torch.LongTensor(target_lengths)

        return source_sentences, target_sentences, source_lengths, target_lengths
