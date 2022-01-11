# from utils  import *
from models import *
from train import SAVE_TO_LOG, PATH_TO_SENTENCEPIECE_MODEL

def greedy_decode(model, src_sentence, src_mask, max_len, start_symbol):
    src_sentence = src_sentence.to(DEVICE)
    src_mask     = src_mask.to(DEVICE)

    memory = model.encode(src_sentence, src_mask)
    ys     = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    for i in range(max_len - 1):
        memory   = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)

        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        
        prob         = model.linear_transform(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word    = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src_sentence.data).fill_(next_word)], dim = 0)
        
        if next_word == EOS_IDX:
            break

    return ys


def translate(model: torch.nn.Module, src_sentences: List[str], CFG):
    model.eval()
    tgt_sentences = []
    sp = spm.SentencePieceProcessor(model_file = os.path.join(PATH_TO_SENTENCEPIECE_MODEL, CFG['sentpiece_model']))
   
    for sentence in src_sentences:
        print(sentence)
        src_sentence_encoded = sp.encode(sentence.strip(), out_type = int)
        print(src_sentence_encoded)
        src_tensor = torch.tensor(src_sentence_encoded, dtype = torch.long)
        #src_tensor =  torch.cat(
        #        (torch.tensor([BOS_IDX]), src_tensor, torch.tensor([EOS_IDX])), dim = 0
        #    )
        print(src_tensor.shape)
        num_tokens = src_tensor.shape[0]

        src_mask    = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens  = greedy_decode(model, src_tensor, src_mask, max_len = num_tokens + 5, start_symbol = BOS_IDX).flatten()
        tgt_tokens  = list(tgt_tokens.cpu().numpy())
        tgt_tokens  = [int(token) for token in tgt_tokens]
        tgt_decoded = sp.decode(tgt_tokens)
        tgt_sentences.append("".join(tgt_decoded))
    
    return tgt_sentences


if __name__ == "__main__":
    CFG = {
        'id': 0,
        'batch_size_t': 1,
        'batch_size_v': 1,
        
        # Optimizer Hyper-parameters
        'learning_rate': 0.0001,
        'betas': (0.9, 0.98),
        'eps': 1e-9,

        # Vocabulary Hyper-parameters
        'src_vocab_size':  32000,
        'tgt_vocab_size':  32000, 
        'sentpiece_model': 'sentpiece_32k.model',

        # Architecture Hyper-parameters
        'architecture_type': 'transformer',
        'embedding_size': 512,               # transformer & rnn

        'n_heads': 8,                        # transformer
        'ffn_hidden_dim': 2048,              # transformer
        'num_encoder_layers': 2,             # transformer
        'num_decoder_layers': 2,             # transformer

        'attention_dim': 8,                  # rnn
        'encoder_hidden_dim': 64,            # rnn
        'decoder_hidden_dim': 64,            # rnn
        'encoder_dropout': 0.5,              # rnn
        'decoder_dropout': 0.5,              # rnn

        # Training Script Parameters
        'epochs': 10,
        'num_workers': 4,
        'debug': True, 
        'print_freq': 100, 
        'observation': None, # "Should be a string, more specific information for experiments"
    }

    LOSS = "1.00"
    PATH_TO_MODELS = os.path.join(PATH_TO_MODELS, 'model-{}'.format(CFG['id']))
    PATH_TO_BEST_MODEL = os.path.join(PATH_TO_MODELS, 'model_{}_name_{}_loss_{}.pth'.format(CFG["id"], CFG["architecture_type"], LOSS))
    
    model = Seq2SeqTransformer(
            CFG['num_encoder_layers'], 
            CFG['num_decoder_layers'], 
            CFG['embedding_size'], 
            CFG['n_heads'], 
            CFG['src_vocab_size'], 
            CFG['tgt_vocab_size'], 
            CFG['ffn_hidden_dim']
    ).to(DEVICE)

    print('aici')
    states = torch.load(PATH_TO_BEST_MODEL, map_location = torch.device('cpu'))
    model.load_state_dict(states['model'])       
    model.eval() 

    with open(PATH_TO_CLEANED_VALID[SRC_LANGUAGE], 'r') as src_file: valid_src_sentences = src_file.read().splitlines()
    with open(PATH_TO_CLEANED_VALID[TGT_LANGUAGE], 'r') as tgt_file: valid_tgt_sentences = tgt_file.read().splitlines()

    test_sentences  = valid_src_sentences[:10]
    label_sentences = valid_tgt_sentences[:10]

    tgt_sentences = translate(model, test_sentences)

    test_sentences  = valid_src_sentences[2:10]
    label_sentences = valid_tgt_sentences[2:10]

    tgt_sentences = translate(model, test_sentences, CFG)
    
    for (predicted_sentence, label_sentence) in zip(tgt_sentences, label_sentences):
        print("Predicted: ", predicted_sentence)
        print("Correct: ",   label_sentence)
        print('\n\n\n')

    hypothesis_corpus = tgt_sentences
    references_corpus = label_sentences

    print('Cumulative 1-gram: {}'.format(corpus_bleu(references_corpus, hypothesis_corpus, weights = (1, 0, 0, 0)) * 100))
    print('Cumulative 2-gram: {}'.format(corpus_bleu(references_corpus, hypothesis_corpus, weights = (0.5, 0.5, 0, 0)) * 100))
    print('Cumulative 3-gram: {}'.format(corpus_bleu(references_corpus, hypothesis_corpus, weights = (0.33, 0.33, 0.33, 0)) * 100))
    print('Cumulative 4-gram: {}'.format(corpus_bleu(references_corpus, hypothesis_corpus, weights = (0.25, 0.25, 0.25, 0.25)) * 100))
