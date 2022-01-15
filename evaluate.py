from models import *
from dataset import *
from train import CFG
from translate import translate


test_loader = SequenceLoader(
    data_folder     = PATH_TO_DATASET,
    vocab_size      = 37000, 
    source_language = "es",
    target_language = "ro",
    dataset_type    = "dev",
    tokens_in_batch = None
)

test_loader.create_batches()

bpe_model = youtokentome.BPE(model = os.path.join(PATH_TO_DATASET, 'bpe_37000.model'))
model     = get_model(CFG).to(DEVICE)

checkpoint = torch.load("model_0_name_transformer_loss_7.33.pth")
model.load_state_dict(checkpoint['model'])
model.eval()

with torch.no_grad():
    hypotheses = list()
    references = list()
    for i, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(
            tqdm(test_loader, total = test_loader.n_batches)):
        hypotheses.append(translate(source_sequence=source_sequence, model = model,
                                    beam_size=4,
                                    length_norm_coefficient=0.6)[0])

        references.extend(test_loader.bpe_model.decode(target_sequence.tolist(), ignore_ids=[0, 2, 3]))

    # for idx, (hypothesis, reference) in enumerate(zip(hypotheses, references)):
    #     print("Predicted: ", hypotheses)
    #     print("Target: ", reference)
    #     print("\n\n\n")
        
    #     if idx == 15: break

    print("\n13a tokenization, cased:\n")
    print(sacrebleu.corpus_bleu(hypotheses, [references]))
    print("\n13a tokenization, caseless:\n")
    print(sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True))
    print("\nInternational tokenization, cased:\n")
    print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl'))
    print("\nInternational tokenization, caseless:\n")
    print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl', lowercase=True))
    print("\n")