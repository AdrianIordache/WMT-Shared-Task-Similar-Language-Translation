from utils import *

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
