from train import CFG
from models import *
from config_file import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', 
    dest = 'gpu', type = int, 
    default = 0, help = "GPU enable for running the procces"
)

args   = parser.parse_args()
RANK   = args.gpu
DEVICE = torch.device(f'cuda:{RANK}' if torch.cuda.is_available() else 'cpu')

def translate(source_sequence, model, bpe_model, beam_size = 4, length_norm_coefficient = 0.6):
    with torch.no_grad():
        k = beam_size

        # Minimum number of hypotheses to complete
        n_completed_hypotheses = min(k, 10)

        # Vocab size
        vocab_size = bpe_model.vocab_size()

        # If the source sequence is a string, convert to a tensor of IDs
        if isinstance(source_sequence, str):
            encoder_sequences = bpe_model.encode(source_sequence, output_type = youtokentome.OutputType.ID, bos = False, eos = False)
            encoder_sequences = torch.LongTensor(encoder_sequences).unsqueeze(0)  # (1, source_sequence_length)
        else:
            encoder_sequences = source_sequence

        # (1, source_sequence_length)
        encoder_sequences        = encoder_sequences.to(DEVICE)  
        encoder_sequence_lengths = torch.LongTensor([encoder_sequences.size(1)]).to(DEVICE)

        # (1, source_sequence_length, d_model)
        encoder_sequences = model.encoder(
            encoder_sequences        = encoder_sequences, 
            encoder_sequence_lengths = encoder_sequence_lengths
        )

        # Our hypothesis to begin with is just <BOS>
        hypotheses = torch.LongTensor([[bpe_model.subword_to_id('<BOS>')]]).to(DEVICE)  # (1, 1)
        hypotheses_lengths = torch.LongTensor([hypotheses.size(1)]).to(DEVICE)  # (1)

        # Tensor to store hypotheses' scores; now it's just 0
        hypotheses_scores = torch.zeros(1).to(DEVICE)  # (1)

        completed_hypotheses        = list()
        completed_hypotheses_scores = list()

        step = 1
        # Assume "s" is the number of incomplete hypotheses currently in the bag; a number less than or equal to "k"
        # At this point, s is 1, because we only have 1 hypothesis to work with, i.e. "<BOS>"
        while True:
            s = hypotheses.size(0)
            
            # (s, step, vocab_size)
            decoder_sequences = model.decoder(
                decoder_sequences        = hypotheses,
                decoder_sequence_lengths = hypotheses_lengths,
                encoder_sequences        = encoder_sequences.repeat(s, 1, 1),
                encoder_sequence_lengths = encoder_sequence_lengths.repeat(s)
            ) 

            # Scores at this step
            scores = decoder_sequences[:, -1, :]      # (s, vocab_size)
            scores = F.log_softmax(scores, dim = -1)  # (s, vocab_size)

            # Add hypotheses' scores from last step to scores at this step to get scores for all possible new hypotheses
            scores = hypotheses_scores.unsqueeze(1) + scores  # (s, vocab_size)

            # Unroll and find top k scores, and their unrolled indices
            top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(k, 0, True, True)  # (k)

            # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
            prev_word_indices = unrolled_indices // vocab_size  # (k)
            next_word_indices = unrolled_indices % vocab_size  # (k)

            # Construct the the new top k hypotheses from these indices
            top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)], dim=1)  # (k, step + 1)

            # Which of these new hypotheses are complete (reached <EOS>)?
            complete = next_word_indices == bpe_model.subword_to_id('<EOS>')  # (k), bool

            # Set aside completed hypotheses and their scores normalized by their lengths
            # For the length normalization formula, see
            # "Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"
            completed_hypotheses.extend(top_k_hypotheses[complete].tolist())
            norm = math.pow(((5 + step) / (5 + 1)), length_norm_coefficient)
            completed_hypotheses_scores.extend((top_k_hypotheses_scores[complete] / norm).tolist())

            # Stop if we have completed enough hypotheses
            if len(completed_hypotheses) >= n_completed_hypotheses:
                break

            # Else, continue with incomplete hypotheses
            hypotheses = top_k_hypotheses[~complete]  # (s, step + 1)
            hypotheses_scores = top_k_hypotheses_scores[~complete]  # (s)
            hypotheses_lengths = torch.LongTensor(hypotheses.size(0) * [hypotheses.size(1)]).to(DEVICE)  # (s)

            # Stop if things have been going on for too long
            if step > 100:
                break

            step += 1

        # If there is not a single completed hypothesis, use partial hypotheses
        if len(completed_hypotheses) == 0:
            completed_hypotheses = hypotheses.tolist()
            completed_hypotheses_scores = hypotheses_scores.tolist()

        # Decode the hypotheses
        all_hypotheses = list()
        for i, h in enumerate(bpe_model.decode(completed_hypotheses, ignore_ids=[0, 2, 3])):
            all_hypotheses.append({"hypothesis": h, "score": completed_hypotheses_scores[i]})

        # Find the best scoring completed hypothesis
        i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
        best_hypothesis = all_hypotheses[i]["hypothesis"]

        return best_hypothesis, all_hypotheses

if __name__ == "__main__":
    DEBUG         = None
    DATASET_TYPE  = "test"
    PATH_TO_MODEL = "models/dataset-3/adrian/gpu-1/model-2/model_2_name_transformer_loss_2.79.pth"
    PATH_TO_PREDS = "logs/predictions/"

    USER       = PATH_TO_MODEL.split("/")[2]
    GPU        = PATH_TO_MODEL.split("/")[3].split("-")[-1]
    MODEL_NAME = PATH_TO_MODEL.split("/")[-1][:-4]

    if os.path.isdir(PATH_TO_PREDS) == False: os.makedirs(PATH_TO_PREDS)

    with open(PATH_TO_DATASET_FILES[DATASET_TYPE][SRC_LANGUAGE], "r", encoding = "utf-8") as src_file:
        src_sentences = src_file.read().splitlines()

    with open(PATH_TO_DATASET_FILES[DATASET_TYPE][TGT_LANGUAGE], "r", encoding = "utf-8") as tgt_file:
        tgt_sentences = tgt_file.read().splitlines()

    if DEBUG is not None:
        src_sentences = src_sentences[: DEBUG]
        tgt_sentences = tgt_sentences[: DEBUG]

    model     = get_model(CFG, DEVICE).to(DEVICE)
    bpe_model = youtokentome.BPE(model = PATH_TO_BPE_MODEL)

    states = torch.load(PATH_TO_MODEL, map_location = torch.device('cpu'))
    model.load_state_dict(states['model'])
    model.eval()

    best_hypotheses = []
    for sentence in src_sentences:
        best_hypothesis, all_hypotheses = translate(sentence, model, bpe_model)
        best_hypotheses.append(best_hypothesis)

    #for item, (predicted_sentence, reference_sentence) in enumerate(zip(best_hypotheses, tgt_sentences)):
    #    print("Predicted: ", predicted_sentence)
    #    print('\n')
    #    print("Reference: ", reference_sentence)
    #    print('\n\n\n')

    with open(os.path.join(PATH_TO_PREDS, f'{DATASET_TYPE}_user_{USER}_gpu_{GPU}_{MODEL_NAME}.out'),'w') as file:
        file.write('\n'.join(best_hypotheses))

    # print('Cumulative 1-gram: {}'.format(corpus_bleu(tgt_sentences, best_hypotheses, weights = (1, 0, 0, 0)) * 100))
    # print('Cumulative 2-gram: {}'.format(corpus_bleu(tgt_sentences, best_hypotheses, weights = (0.5, 0.5, 0, 0)) * 100))
    # print('Cumulative 3-gram: {}'.format(corpus_bleu(tgt_sentences, best_hypotheses, weights = (0.33, 0.33, 0.33, 0)) * 100))
    # print('Cumulative 4-gram: {}'.format(corpus_bleu(tgt_sentences, best_hypotheses, weights = (0.25, 0.25, 0.25, 0.25)) * 100))

    print("\n13a tokenization, cased:\n")
    print(sacrebleu.corpus_bleu(best_hypotheses, [tgt_sentences]))
    print("\n13a tokenization, caseless:\n")
    print(sacrebleu.corpus_bleu(best_hypotheses, [tgt_sentences], lowercase=True))
    print("\nInternational tokenization, cased:\n")
    print(sacrebleu.corpus_bleu(best_hypotheses, [tgt_sentences], tokenize='intl'))
    print("\nInternational tokenization, caseless:\n")
    print(sacrebleu.corpus_bleu(best_hypotheses, [tgt_sentences], tokenize='intl', lowercase=True))
    print("\n")
